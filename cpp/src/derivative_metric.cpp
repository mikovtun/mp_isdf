#include "derivative_metric.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {
using Basis = std::array<int,2>;
struct Shell { FINT id; std::size_t offset, size; };
struct Kernel {
    LibcintWrapper::Integral kind;
    std::array<int,4> permutation; // canonical AO slot -> requested AO slot
    int component;
};

Kernel kernel_for(Basis left, Basis right)
{
    const std::array<int,4> labels{left[0],left[1],right[0],right[1]};
    int count = 0;
    for (int v : labels) {
        if (v < 0 || v > 3) throw std::invalid_argument("invalid AO derivative label");
        count += (v != 0);
    }
    LibcintWrapper::Integral kind;
    std::array<int,4> wanted{};
    if (count == 0) kind = LibcintWrapper::Integral::int2e;
    else if (count == 1) {
        kind = LibcintWrapper::Integral::int2e_ip1;
        wanted[0] = 1;
    } else if (count == 2) {
        if ((left[0] && left[1]) || (right[0] && right[1])) {
            kind = LibcintWrapper::Integral::int2e_ipvip1;
            wanted[0] = wanted[1] = 1;
        } else {
            kind = LibcintWrapper::Integral::int2e_ip1ip2;
            wanted[0] = wanted[2] = 1;
        }
    } else if (count == 4) {
        kind = LibcintWrapper::Integral::int2e_ipvip1ipvip2;
        wanted.fill(1);
    } else throw std::invalid_argument("three-slot derivative ERIs are unavailable in libcint");
    for (int exchange = 0; exchange < 2; ++exchange)
        for (int flip_left = 0; flip_left < 2; ++flip_left)
            for (int flip_right = 0; flip_right < 2; ++flip_right) {
                std::array<int,4> perm = exchange ? std::array<int,4>{2,3,0,1}
                                                  : std::array<int,4>{0,1,2,3};
                if (flip_left) std::swap(perm[0],perm[1]);
                if (flip_right) std::swap(perm[2],perm[3]);
                bool matches = true;
                int component = 0;
                for (int i=0;i<4;++i) {
                    matches &= (labels[perm[i]] != 0) == (wanted[i] != 0);
                    if (wanted[i]) component = 3*component + labels[perm[i]]-1;
                }
                if (matches) return {kind,perm,component};
            }
    throw std::logic_error("unsupported derivative slot permutation");
}

// All indexes below are ordered AO pairs. libcint's first AO index is fastest.
std::vector<double> quartet(const LibcintWrapper& tables, const Kernel& kernel,
                            std::array<FINT,4> shells, std::array<std::size_t,4> sizes)
{
    std::array<FINT,4> ordered_shells{};
    std::size_t count=1;
    for (int i=0;i<4;++i) {
        ordered_shells[i]=shells[kernel.permutation[i]];
        const auto n=sizes[kernel.permutation[i]];
        if (n && count > std::numeric_limits<std::size_t>::max()/n)
            throw std::overflow_error("derivative ERI quartet too large");
        count *= n;
    }
    const auto components=static_cast<std::size_t>(LibcintWrapper::components(kernel.kind));
    if (count > std::numeric_limits<std::size_t>::max()/components)
        throw std::overflow_error("derivative ERI components too large");
    std::vector<double> out(count*components,0.0);
    tables.evaluate(kernel.kind,out.data(),tables.dimensions(ordered_shells),ordered_shells);
    return out;
}

std::size_t value_index(const Kernel& kernel, std::array<std::size_t,4> dims,
                        std::array<std::size_t,4> indices)
{
    std::size_t idx=0, stride=1;
    for (int i=0;i<4;++i) {
        idx += stride*indices[kernel.permutation[i]];
        stride *= dims[kernel.permutation[i]];
    }
    return static_cast<std::size_t>(kernel.component)*stride+idx;
}
}

std::vector<double> derivative_cross_metric(
    const LibcintContext& ctx, const std::vector<double>& left, std::size_t nleft,
    Basis left_basis, const std::vector<double>& right, std::size_t nright,
    Basis right_basis, double tolerance, std::size_t max_rank)
{
    if (!std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument("Cholesky tolerance must be finite and nonnegative");
    const auto nao=ctx.nao();
    if (!nao || nao > std::numeric_limits<std::size_t>::max()/nao/2)
        throw std::overflow_error("joint AO-pair dimension overflow");
    const auto npair=nao*nao, dimension=2*npair;
    if (nleft > std::numeric_limits<std::size_t>::max()/npair ||
        nright > std::numeric_limits<std::size_t>::max()/npair ||
        left.size()!=nleft*npair || right.size()!=nright*npair)
        throw std::invalid_argument("invalid derivative channel coefficient dimensions");
    if (max_rank > dimension) throw std::invalid_argument("metric rank exceeds joint pair dimension");
    for (double c:left) if (!std::isfinite(c)) throw std::invalid_argument("nonfinite left coefficients");
    for (double c:right) if (!std::isfinite(c)) throw std::invalid_argument("nonfinite right coefficients");
    const std::array<Basis,2> bases{left_basis,right_basis};
    std::array<std::array<Kernel,2>,2> kernels{};
    for (int i=0;i<2;++i) for (int j=0;j<2;++j)
        kernels[i][j]=kernel_for(bases[i],bases[j]);
    std::vector<Shell> shells;
    std::vector<std::size_t> ao_shell(nao);
    for (FINT s=0;s<ctx.tables().nbas();++s) {
        const auto size=static_cast<std::size_t>(CINTcgto_spheric(s,ctx.tables().bas_data()));
        const auto offset=shells.empty()?0:shells.back().offset+shells.back().size;
        shells.push_back({s,offset,size});
        for (std::size_t i=offset;i<offset+size;++i) ao_shell.at(i)=static_cast<std::size_t>(s);
    }
    if (shells.back().offset+shells.back().size!=nao)
        throw std::runtime_error("AO and libcint shell ordering mismatch");
    const auto& tables=ctx.tables();
    std::vector<double> diagonal(dimension,0.0);
    for (int channel=0;channel<2;++channel)
        for (const auto& a:shells) for (const auto& b:shells) {
            const auto dims=std::array<std::size_t,4>{a.size,b.size,a.size,b.size};
            const auto block=quartet(tables,kernels[channel][channel],
                                     {a.id,b.id,a.id,b.id},dims);
            for (std::size_t i=0;i<a.size;++i) for (std::size_t j=0;j<b.size;++j) {
                const auto value=block[value_index(kernels[channel][channel],dims,{i,j,i,j})];
                if (!std::isfinite(value) || value < -1e-10)
                    throw std::runtime_error("invalid derivative ERI diagonal");
                diagonal[channel*npair+(a.offset+i)*nao+b.offset+j]=std::max(0.0,value);
            }
        }
    const auto initial=*std::max_element(diagonal.begin(),diagonal.end());
    const double cutoff=tolerance*initial;
    std::vector<double> factors; // [pivot, joint ordered pair]
    std::vector<std::size_t> pivots;
    const auto cap=max_rank?max_rank:dimension;
    while (pivots.size()<cap) {
        const auto it=std::max_element(diagonal.begin(),diagonal.end());
        const double pivot_value=*it;
        if (pivot_value<=cutoff || pivot_value<=0) break;
        const auto pivot=static_cast<std::size_t>(it-diagonal.begin());
        const auto pivot_channel=pivot/npair;
        const auto mu=(pivot%npair)/nao, nu=(pivot%npair)%nao;
        const auto& a=shells[ao_shell[mu]];
        const auto& b=shells[ao_shell[nu]];
        std::vector<double> column(dimension,0.0);
        for (int channel=0;channel<2;++channel)
            for (const auto& c:shells) for (const auto& d:shells) {
                const auto dims=std::array<std::size_t,4>{c.size,d.size,a.size,b.size};
                const auto& kernel=kernels[channel][pivot_channel];
                const auto block=quartet(tables,kernel,{c.id,d.id,a.id,b.id},dims);
                for (std::size_t i=0;i<c.size;++i) for (std::size_t j=0;j<d.size;++j)
                    column[channel*npair+(c.offset+i)*nao+d.offset+j]=
                        block[value_index(kernel,dims,{i,j,mu-a.offset,nu-b.offset})];
            }
        const auto rank=pivots.size();
        const auto denom=std::sqrt(pivot_value);
        for (std::size_t p=0;p<dimension;++p) {
            double value=column[p];
            for (std::size_t k=0;k<rank;++k)
                value-=factors[k*dimension+p]*factors[k*dimension+pivot];
            column[p]=value/denom;
            if (!std::isfinite(column[p])) throw std::runtime_error("nonfinite derivative Cholesky vector");
        }
        if (dimension > std::numeric_limits<std::size_t>::max()/(rank+1))
            throw std::overflow_error("derivative Cholesky factors too large");
        factors.insert(factors.end(),column.begin(),column.end());
        pivots.push_back(pivot);
        for (std::size_t p=0;p<dimension;++p)
            diagonal[p]=std::max(0.0,diagonal[p]-column[p]*column[p]);
        diagonal[pivot]=0.0;
    }
    const auto rank=pivots.size();
    if (rank && (nleft > std::numeric_limits<std::size_t>::max()/rank ||
                 nright > std::numeric_limits<std::size_t>::max()/rank))
        throw std::overflow_error("derivative projections too large");
    if (nleft && nright > std::numeric_limits<std::size_t>::max()/nleft)
        throw std::overflow_error("derivative metric too large");
    std::vector<double> metric(nleft*nright,0.0);
    std::vector<double> projected_left(nleft*rank,0.0), projected_right(nright*rank,0.0);
    for (std::size_t k=0;k<rank;++k) {
        for (std::size_t v=0;v<nleft;++v)
            for (std::size_t p=0;p<npair;++p)
                projected_left[v*rank+k]+=left[v*npair+p]*factors[k*dimension+p];
        for (std::size_t v=0;v<nright;++v)
            for (std::size_t p=0;p<npair;++p)
                projected_right[v*rank+k]+=right[v*npair+p]*factors[k*dimension+npair+p];
    }
    for (std::size_t v=0;v<nleft;++v) for (std::size_t w=0;w<nright;++w)
        for (std::size_t k=0;k<rank;++k)
            metric[v*nright+w]+=projected_left[v*rank+k]*projected_right[w*rank+k];
    return metric;
}
}
