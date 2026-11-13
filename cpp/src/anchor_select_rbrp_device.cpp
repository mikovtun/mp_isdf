#include "anchor_select_rbrp_device.hpp"

#ifdef ISDF_ENABLE_CUDA
#include <magma_v2.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {
using Space = Kokkos::Cuda;
using View = Kokkos::View<double*, Space>;
using Flags = Kokkos::View<int*, Space>;
using Policy = Kokkos::RangePolicy<Space>;

void check(magma_int_t info, const char* operation) {
    if (info != 0) throw std::runtime_error(std::string("RBRP MAGMA ")+operation+" failed: "+std::to_string(info));
}
struct MagmaContext {
    magma_queue_t queue = nullptr;
    MagmaContext() {
        if (magma_init() != MAGMA_SUCCESS) throw std::runtime_error("magma_init failed");
        magma_device_t device;
        magma_getdevice(&device);
        magma_queue_create(device, &queue);
    }
    ~MagmaContext() { if (queue) magma_queue_destroy(queue); magma_finalize(); }
    MagmaContext(const MagmaContext&) = delete;
    MagmaContext& operator=(const MagmaContext&) = delete;
    void sync() const { magma_queue_sync(queue); }
};

// AOs and packed pair rows are generated in CUDA space. Columns of the pair
// matrix are padded to lda for MAGMA's GPU QR workspace and BLAS alignment.
class PairStream {
public:
    PairStream(const AnchorSelection& selector,
               const std::vector<std::array<double,3>>& points,
               const std::vector<double>& weights,
               std::size_t batch, std::size_t lda)
        : selector_(selector), points_(points), weights_(weights), batch_(batch),
          nao_(selector.nao()), p_(nao_*(nao_+1)/2), lda_(lda) {}
    std::size_t n() const { return points_.size(); }
    std::size_t nao() const { return nao_; }
    std::size_t p() const { return p_; }
    std::size_t lda() const { return lda_; }
    double weight(std::size_t g) const { return weights_[g]; }
    void batches(const auto& fn) const {
        for (std::size_t start = 0; start < n(); start += batch_) {
            const auto count = std::min(batch_, n()-start);
            std::vector<std::size_t> ids(count);
            std::iota(ids.begin(), ids.end(), start);
            fn(start, count, rows(ids));
        }
    }
    View rows(const std::vector<std::size_t>& ids) const {
        std::vector<std::array<double,3>> coords;
        coords.reserve(ids.size());
        View w("rbrp_batch_sqrtw", ids.size());
        auto hw = Kokkos::create_mirror_view(w);
        for (std::size_t j=0; j<ids.size(); ++j) {
            coords.push_back(points_[ids[j]]);
            hw(j) = std::sqrt(std::abs(weights_[ids[j]]));
        }
        Kokkos::deep_copy(w, hw);
        auto axis = selector_.evaluate_axis<Space>(coords, {AxisType::AtomicOrbital});
        auto ao = axis.values;
        const auto nao = nao_, p = p_, lda = lda_;
        View out("rbrp_packed_pair_batch", ids.size()*lda);
        Kokkos::parallel_for("rbrp_pack_pairs", Policy(0,ids.size()*lda),
            KOKKOS_LAMBDA(const std::size_t x) {
                const std::size_t g=x/lda, t=x%lda;
                if (t >= p) { out(x)=0; return; }
                // Upper triangle in row-major order, with sqrt(2) off diagonal.
                std::size_t mu=0, first=0;
                while (first+nao-mu <= t) { first+=nao-mu; ++mu; }
                const std::size_t nu=mu+t-first;
                out(x) = w(g)*ao(g*nao+mu)*ao(g*nao+nu)*
                    (mu == nu ? 1.0 : 1.4142135623730950488);
            });
        Kokkos::fence();
        return out;
    }
    View initial_norms() const {
        View d("rbrp_residual_norms", n());
        for (std::size_t start=0; start<n(); start+=batch_) {
            const auto count=std::min(batch_,n()-start);
            std::vector<std::array<double,3>> coords(points_.begin()+start,
                points_.begin()+start+count);
            auto axis=selector_.evaluate_axis<Space>(coords,{AxisType::AtomicOrbital});
            auto ao=axis.values;
            View w("rbrp_initial_weights", count);
            auto hw=Kokkos::create_mirror_view(w);
            for (std::size_t j=0; j<count; ++j) hw(j)=std::abs(weights_[start+j]);
            Kokkos::deep_copy(w,hw);
            const auto nao=nao_;
            Kokkos::parallel_for("rbrp_initial_norm",Policy(0,count),
                KOKKOS_LAMBDA(const std::size_t j) {
                    double norm=0;
                    for (std::size_t a=0; a<nao; ++a)
                        norm+=ao(j*nao+a)*ao(j*nao+a);
                    d(start+j)=w(j)*norm*norm;
                });
        }
        Kokkos::fence();
        return d;
    }
private:
    const AnchorSelection& selector_;
    const std::vector<std::array<double,3>>& points_;
    const std::vector<double>& weights_;
    std::size_t batch_,nao_,p_,lda_;
};

double sum(View d, std::size_t n) {
    double v=0;
    Kokkos::parallel_reduce("rbrp_sum",Policy(0,n),
        KOKKOS_LAMBDA(const std::size_t i,double& acc){acc+=d(i);},v);
    return v;
}

// GPU scan and threshold selection; only the total and selected index return
// to the host. Mask is temporary within a candidate block.
std::size_t sample(View d, Flags mask, View prefix, std::size_t n,
                   std::mt19937_64& rng) {
    Kokkos::parallel_scan("rbrp_candidate_scan",Policy(0,n),
        KOKKOS_LAMBDA(const std::size_t i,double& update,const bool final) {
            update += mask(i) ? 0.0 : d(i);
            if (final) prefix(i)=update;
        });
    auto last=Kokkos::subview(prefix,n-1);
    double total=0;
    Kokkos::deep_copy(total,last);
    if (!(total>0) || !std::isfinite(total)) return n;
    std::uniform_real_distribution<double> uniform(0.0,total);
    const double target=uniform(rng);
    std::size_t id=n;
    Kokkos::parallel_reduce("rbrp_choose",Policy(0,n),
        KOKKOS_LAMBDA(const std::size_t i,std::size_t& best) {
            if (!mask(i) && d(i)>0 && prefix(i)>target && i<best) best=i;
        },Kokkos::Min<std::size_t>(id));
    // Rounding at the far end of the cumulative distribution: take the last
    // positive-weight candidate rather than terminating early.
    if (id==n) {
        Kokkos::parallel_reduce("rbrp_last_candidate",Policy(0,n),
            KOKKOS_LAMBDA(const std::size_t i,std::size_t& best) {
                if (!mask(i) && d(i)>0 && i<best) best=i;
            },Kokkos::Min<std::size_t>(id));
    }
    if (id<n) Kokkos::parallel_for("rbrp_mask_sample",Policy(id,id+1),
        KOKKOS_LAMBDA(const std::size_t i){mask(i)=1;});
    return id;
}

// MAGMA BLAS operates on CUDA pointers; Kokkos and MAGMA queues are fenced
// at each boundary, not assumed to share a stream.
void project(View v, View q, View coefficients, std::size_t lda,
             std::size_t p, std::size_t old, std::size_t count,
             const MagmaContext& magma) {
    if (!old) return;
    Kokkos::fence();
    for (int pass=0; pass<2; ++pass) {
        magma_dgemm(MagmaTrans,MagmaNoTrans,old,count,p,1.0,
            q.data(),lda,v.data(),lda,0.0,coefficients.data(),old,magma.queue);
        magma_dgemm(MagmaNoTrans,MagmaNoTrans,p,count,old,-1.0,
            q.data(),lda,coefficients.data(),old,1.0,v.data(),lda,magma.queue);
    }
    magma.sync();
}

// Explicit X - LQ^T refresh, with batchwise MAGMA GEMM.
void refresh(const PairStream& op, View q, View l, View d,
             Flags selected, std::size_t rank, const MagmaContext& magma) {
    const auto n=op.n(), p=op.p(), lda=op.lda();
    op.batches([&](std::size_t start,std::size_t count,View x) {
        if (rank) {
            // L is row-major [n,rank], i.e., column-major [rank,n].
            magma_dgemm(MagmaNoTrans,MagmaNoTrans,p,count,rank,-1.0,
                q.data(),lda,l.data()+start*rank,rank,1.0,x.data(),lda,magma.queue);
            magma.sync();
        }
        Kokkos::parallel_for("rbrp_refresh_norms",Policy(0,count),
            KOKKOS_LAMBDA(const std::size_t g) {
                if (selected(start+g)) { d(start+g)=0; return; }
                double norm=0;
                for (std::size_t t=0;t<p;++t) norm+=x(g*lda+t)*x(g*lda+t);
                d(start+g)=norm;
            });
        Kokkos::fence();
    });
}

void append(const PairStream& op, View qb, View& q, View& l, View d,
            Flags selected, std::size_t old, std::size_t keep,
            const MagmaContext& magma) {
    const auto n=op.n(), p=op.p(), lda=op.lda(), rank=old+keep;
    View grown_q("rbrp_device_Q",lda*rank);
    Kokkos::parallel_for("rbrp_copy_Q",Policy(0,lda*rank),
        KOKKOS_LAMBDA(const std::size_t i) {
            grown_q(i)=i<lda*old ? q(i) : qb(i-lda*old);
        });
    View grown_l("rbrp_device_L",n*rank);
    Kokkos::parallel_for("rbrp_copy_L",Policy(0,n*old),
        KOKKOS_LAMBDA(const std::size_t i) {
            grown_l((i/old)*rank+i%old)=l(i);
        });
    Kokkos::fence();
    op.batches([&](std::size_t start,std::size_t count,View x) {
        magma_dgemm(MagmaTrans,MagmaNoTrans,keep,count,p,1.0,
            qb.data(),lda,x.data(),lda,0.0,grown_l.data()+start*rank+old,rank,magma.queue);
        // Leading dimension of the strided L submatrix is rank, but MAGMA
        // writes only keep rows beginning at offset old of each column.
        magma.sync();
        Kokkos::parallel_for("rbrp_update_residual",Policy(0,count),
            KOKKOS_LAMBDA(const std::size_t g) {
                const auto id=start+g;
                if (selected(id)) {
                    for (std::size_t t=0;t<keep;++t) grown_l(id*rank+old+t)=0;
                    d(id)=0;
                    return;
                }
                double decrease=0;
                for (std::size_t t=0;t<keep;++t) {
                    const double v=grown_l(id*rank+old+t);
                    decrease+=v*v;
                }
                d(id)=Kokkos::max(0.0,d(id)-decrease);
            });
        Kokkos::fence();
    });
    q=grown_q; l=grown_l;
}

// MAGMA SVD is a hybrid routine with host small-matrix input; only L1 is
// gathered from device, never L itself.
void finish_id(const PairStream& op, View q, View l, const MagmaContext& magma,
               RBRPResult& out, double svd_rtol) {
    const auto n=op.n(), p=op.p(), lda=op.lda(), k=out.indices.size(), nao=op.nao();
    if (!k) return;
    View d_l1("rbrp_device_L1",k*k);
    View d_index("rbrp_device_indices",k);
    auto hi=Kokkos::create_mirror_view(d_index);
    for (std::size_t a=0;a<k;++a) hi(a)=static_cast<double>(out.indices[a]);
    Kokkos::deep_copy(d_index,hi);
    Kokkos::parallel_for("rbrp_gather_L1",Policy(0,k*k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const auto row=i%k, col=i/k;
            d_l1(i)=l(static_cast<std::size_t>(d_index(row))*k+col);
        });
    auto h_l1=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},d_l1);
    std::vector<double> a(h_l1.data(),h_l1.data()+k*k),s(k),u(k*k),vt(k*k);
    std::vector<magma_int_t> iwork(8*k);
    double query=0;
    magma_int_t info=0;
    magma_dgesdd(MagmaAllVec,k,k,a.data(),k,s.data(),u.data(),k,
        vt.data(),k,&query,-1,iwork.data(),&info);
    check(info,"gesdd workspace query");
    const auto lwork=static_cast<magma_int_t>(std::ceil(query));
    if (lwork<=0) throw std::runtime_error("RBRP MAGMA gesdd invalid workspace");
    std::vector<double> work(lwork);
    magma_dgesdd(MagmaAllVec,k,k,a.data(),k,s.data(),u.data(),k,
        vt.data(),k,work.data(),lwork,iwork.data(),&info);
    check(info,"gesdd");
    out.condition=s.back() ? s.front()/s.back() : std::numeric_limits<double>::infinity();
    for (double val:s) if (val>svd_rtol*s.front()) ++out.retained_rank;
    if (!out.retained_rank) throw std::runtime_error("RBRP anchor block has zero numerical rank");
    // Column-major inv (k,k): same buffer passed to MAGMA GEMM below.
    View inv("rbrp_device_L1_inverse",k*k);
    auto h_inv=Kokkos::create_mirror_view(inv);
    for (std::size_t col=0; col<k; ++col)
        for (std::size_t row=0; row<k; ++row) {
            double value=0;
            for (std::size_t t=0; t<out.retained_rank; ++t)
                value+=vt[t+row*k]*u[col+t*k]/s[t];
            h_inv(row+col*k)=value;
        }
    Kokkos::deep_copy(inv,h_inv);
    View h("rbrp_device_H",k*n);
    // L is [k,n] column-major, inv^T is [k,k] column-major.
    magma_dgemm(MagmaTrans,MagmaNoTrans,k,n,k,1.0,
        inv.data(),k,l.data(),k,0.0,h.data(),k,magma.queue);
    View c("rbrp_device_packed_transform",p*k);
    // Q [lda,k] * inv [k,k], scaled/unpacked below.
    magma_dgemm(MagmaNoTrans,MagmaNoTrans,p,k,k,1.0,
        q.data(),lda,inv.data(),k,0.0,c.data(),p,magma.queue);
    magma.sync();
    View d_theta("rbrp_device_theta",k*n);
    View d_transform("rbrp_device_transform",k*nao*nao);
    View d_w("rbrp_device_weight",n);
    auto hw=Kokkos::create_mirror_view(d_w);
    for (std::size_t g=0;g<n;++g) hw(g)=op.weight(g);
    Kokkos::deep_copy(d_w,hw);
    Kokkos::parallel_for("rbrp_scale_theta",Policy(0,k*n),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t g=i/k, aidx=i%k, anchor=static_cast<std::size_t>(d_index(aidx));
            d_theta(aidx*n+g)=h(i)*std::sqrt(Kokkos::abs(d_w(anchor))) /
                (d_w(anchor)*std::sqrt(Kokkos::abs(d_w(g))));
        });
    Kokkos::parallel_for("rbrp_anchor_identity",Policy(0,k*k),
        KOKKOS_LAMBDA(const std::size_t i) {
            const auto aidx=i/k, b=i%k, anchor=static_cast<std::size_t>(d_index(b));
            d_theta(aidx*n+anchor)=aidx==b ? 1.0/d_w(anchor) : 0.0;
        });
    Kokkos::parallel_for("rbrp_unpack_transform",Policy(0,k*nao*nao),
        KOKKOS_LAMBDA(const std::size_t i) {
            const auto aidx=i/(nao*nao), mu=(i/nao)%nao, nu=i%nao;
            const auto lo=Kokkos::min(mu,nu), hi=Kokkos::max(mu,nu);
            const auto packed=lo*nao-lo*(lo-1)/2+(hi-lo);
            const auto anchor=static_cast<std::size_t>(d_index(aidx));
            const double scale=mu==nu ? 1.0 : 0.7071067811865475244;
            d_transform(i)=c(aidx*p+packed)*scale*
                std::sqrt(Kokkos::abs(d_w(anchor)))/d_w(anchor);
        });
    auto ht=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},d_theta);
    auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},d_transform);
    out.interp_vecs.assign(ht.data(),ht.data()+k*n);
    out.transform.assign(hc.data(),hc.data()+k*nao*nao);
}

} // namespace

RBRPResult rbrp_ao_device(const AnchorSelection& selector,
    const std::vector<std::array<double,3>>& points,
    const std::vector<double>& weights, std::size_t batch_size,
    std::size_t block_size, double tolerance, std::size_t max_rank,
    std::uint64_t seed, double tau_block, double rank_rtol,
    std::size_t refresh_every, double svd_rtol, bool compute_id) {
    const auto n=points.size(), nao=selector.nao();
    if (!n || !nao || !batch_size || !block_size || weights.size()!=n ||
        nao==std::numeric_limits<std::size_t>::max() ||
        nao>std::numeric_limits<std::size_t>::max()/(nao+1) ||
        !std::isfinite(tolerance) || tolerance<0 ||
        !std::isfinite(rank_rtol) || rank_rtol<=0 ||
        !std::isfinite(svd_rtol) || svd_rtol<0 ||
        (std::isfinite(tau_block) && (tau_block<0 || tau_block>=1)) ||
        (!std::isfinite(tau_block) && !std::isnan(tau_block)))
        throw std::invalid_argument("invalid device RBRP dimensions or tolerances");
    for (std::size_t g=0;g<n;++g) {
        if (!std::isfinite(weights[g]) || weights[g]==0)
            throw std::invalid_argument("device RBRP requires finite nonzero weights");
        for (double coordinate:points[g])
            if (!std::isfinite(coordinate))
                throw std::invalid_argument("device RBRP requires finite coordinates");
    }
    const auto p=nao*(nao+1)/2, cap=std::min({p,n,max_rank?max_rank:std::min(n,p)});
    if (p>static_cast<std::size_t>(std::numeric_limits<magma_int_t>::max())-32 ||
        (compute_id && nao>std::numeric_limits<std::size_t>::max()/nao))
        throw std::overflow_error("device RBRP pair width exceeds supported range");
    const auto lda=static_cast<std::size_t>(magma_roundup(p,32));
    if (lda>static_cast<std::size_t>(std::numeric_limits<magma_int_t>::max()) ||
        block_size>static_cast<std::size_t>(std::numeric_limits<magma_int_t>::max()) ||
        std::min(batch_size,n)>std::numeric_limits<std::size_t>::max()/lda ||
        cap>std::numeric_limits<std::size_t>::max()/n ||
        cap>std::numeric_limits<std::size_t>::max()/lda ||
        (compute_id && cap>std::numeric_limits<std::size_t>::max()/(nao*nao)))
        throw std::overflow_error("device RBRP dimensions exceed supported range");
    if (std::isnan(tau_block)) tau_block=block_size>1 ? 1.0/block_size : 0.0;
    MagmaContext magma;
    PairStream op(selector,points,weights,batch_size,lda);
    View d=op.initial_norms();
    Flags selected("rbrp_device_selected",n), mask("rbrp_device_sampling_mask",n);
    Kokkos::deep_copy(selected,0); Kokkos::deep_copy(mask,0);
    View cdf("rbrp_device_cdf",n), q("rbrp_device_empty_Q",0), l("rbrp_device_empty_L",0);
    const double total=sum(d,n), target=tolerance*tolerance*total;
    if (!std::isfinite(total)) throw std::overflow_error("device RBRP non-finite norm");
    std::mt19937_64 rng(seed);
    RBRPResult out; out.status="rank_cap";
    std::size_t iteration=0;
    while (true) {
        const auto old=out.indices.size();
        if (old && (sum(d,n)<=target || (refresh_every && iteration%refresh_every==0)))
            refresh(op,q,l,d,selected,old,magma);
        const auto residual=sum(d,n);
        if (residual<=target) {out.status="tolerance"; break;}
        if (old>=cap) break;
        const auto b=std::min(block_size,cap-old);
        std::vector<std::size_t> candidates;
        for (std::size_t t=0;t<b;++t) {
            const auto id=sample(d,mask,cdf,n,rng);
            if (id==n) break;
            candidates.push_back(id);
        }
        if (candidates.empty()) {out.status="no_candidates";break;}
        const auto count=candidates.size();
        View v=op.rows(candidates);
        double norm_original=0;
        Kokkos::parallel_reduce("rbrp_candidate_original_norm",Policy(0,count*p),
            KOKKOS_LAMBDA(const std::size_t i,double& value) {
                const double x=v((i/p)*lda+i%p);
                value+=x*x;
            },norm_original);
        View coeff("rbrp_candidate_projection",std::max<std::size_t>(1,old*count));
        project(v,q,coeff,lda,p,old,count,magma);
        std::vector<magma_int_t> piv(count,0);
        std::vector<double> tau(count);
        magma_int_t host_bytes=-1,device_bytes=-1,info=0;
        magma_dgeqp3_expert_gpu_work(p,count,v.data(),lda,nullptr,nullptr,
            nullptr,&host_bytes,nullptr,&device_bytes,&info,magma.queue);
        check(info,"geqp3 workspace query");
        if (device_bytes<=0 || host_bytes<0) throw std::runtime_error("RBRP MAGMA QR workspace invalid");
        void* work=nullptr;
        if (magma_malloc(&work,device_bytes)!=MAGMA_SUCCESS)
            throw std::runtime_error("RBRP MAGMA QR workspace allocation failed");
        try {
            magma_dgeqp3_expert_gpu_work(p,count,v.data(),lda,piv.data(),tau.data(),
                nullptr,&host_bytes,work,&device_bytes,&info,magma.queue);
            magma.sync(); check(info,"geqp3");
        } catch (...) {magma_free(work);throw;}
        magma_free(work);
        // Only the upper triangular R entries return to host.
        View d_r("rbrp_block_R",count*count);
        Kokkos::parallel_for("rbrp_extract_R",Policy(0,count*count),
            KOKKOS_LAMBDA(const std::size_t i) {
                const auto row=i%count,col=i/count;
                d_r(i)=row<=col ? v(col*lda+row) : 0.0;
            });
        auto hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},d_r);
        std::vector<double> tails(count,0);
        for (std::size_t row=count;row-- >0;) {
            tails[row]=row+1<count ? tails[row+1]:0;
            for (std::size_t col=row;col<count;++col)
                tails[row]+=hr(col*count+row)*hr(col*count+row);
        }
        if (!(tails[0]>0) || !std::isfinite(tails[0])) {
            out.status="numerical_rank";break;
        }
        std::size_t keep=0;
        for (std::size_t i=0;i<count;++i)
            if (tails[i]>=tau_block*tails[0]) keep=i+1;
        const double floor=rank_rtol*std::sqrt(norm_original);
        for (std::size_t i=0;i<keep;++i)
            if (std::abs(hr(i*count+i))<=floor) {keep=i;break;}
        if (!keep) {out.status="numerical_rank";break;}
        std::vector<std::size_t> accepted(keep);
        for (std::size_t j=0;j<keep;++j) {
            if (piv[j]<1 || static_cast<std::size_t>(piv[j])>count)
                throw std::runtime_error("RBRP MAGMA returned invalid pivot");
            accepted[j]=candidates[static_cast<std::size_t>(piv[j]-1)];
        }
        // A fresh unpivoted MAGMA GPU QR supplies a supported dT for orgqr.
        View qb=op.rows(accepted);
        View small_coeff("rbrp_accepted_projection",std::max<std::size_t>(1,old*keep));
        project(qb,q,small_coeff,lda,p,old,keep,magma);
        const auto nb=magma_get_dgeqrf_nb(p,keep);
        View dt("rbrp_qr_T",(2*keep+magma_roundup(keep,32))*nb);
        std::vector<double> tau_accepted(keep);
        magma_dgeqrf_gpu(p,keep,qb.data(),lda,tau_accepted.data(),dt.data(),&info);
        check(info,"geqrf_gpu");
        magma_dorgqr_gpu(p,keep,keep,qb.data(),lda,tau_accepted.data(),dt.data(),nb,&info);
        check(info,"orgqr_gpu");
        magma.sync();
        // Repeat projection and QR to protect orthogonality against the
        // accumulated basis, as in the host reference.
        project(qb,q,small_coeff,lda,p,old,keep,magma);
        magma_dgeqrf_gpu(p,keep,qb.data(),lda,tau_accepted.data(),dt.data(),&info);
        check(info,"reorthogonalizing geqrf_gpu");
        magma_dorgqr_gpu(p,keep,keep,qb.data(),lda,tau_accepted.data(),dt.data(),nb,&info);
        check(info,"reorthogonalizing orgqr_gpu");
        magma.sync();
        append(op,qb,q,l,d,selected,old,keep,magma);
        for (auto id:accepted) out.indices.push_back(id);
        View d_ids("rbrp_accepted_ids",keep);
        auto hids=Kokkos::create_mirror_view(d_ids);
        for (std::size_t j=0;j<keep;++j) hids(j)=static_cast<double>(accepted[j]);
        Kokkos::deep_copy(d_ids,hids);
        Kokkos::parallel_for("rbrp_commit_samples",Policy(0,keep),
            KOKKOS_LAMBDA(const std::size_t j) {
                const auto id=static_cast<std::size_t>(d_ids(j));
                selected(id)=1; d(id)=0;
            });
        for (auto id:candidates)
            Kokkos::parallel_for("rbrp_reset_sampling_mask",Policy(id,id+1),
                KOKKOS_LAMBDA(const std::size_t i){mask(i)=0;});
        Kokkos::fence();
        out.history.insert(out.history.end(),{
            static_cast<double>(out.indices.size()),static_cast<double>(count),
            static_cast<double>(keep),total ? std::sqrt(sum(d,n)/total):0.0});
        ++iteration;
    }
    refresh(op,q,l,d,selected,out.indices.size(),magma);
    out.relative_residual=total ? std::sqrt(sum(d,n)/total):0;
    if (out.relative_residual<=tolerance) out.status="tolerance";
    else if (out.status=="tolerance") out.status="roundoff_limited";
    if (compute_id) finish_id(op,q,l,magma,out,svd_rtol);
    return out;
}
} // namespace isdf
#endif
