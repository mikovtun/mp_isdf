#pragma once

// libcint's generated header does not provide C++ linkage guards.
extern "C" {
#include <cint.h>
}

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// libcint's generated integrals are not declared by cint.h. Declare only the
// real-valued 4-center entry points used here, with their C ABI signatures.
extern "C" {
#define ISDF_DECLARE_2E(name) \
    CACHE_SIZE_T name(double*, FINT*, FINT*, FINT*, FINT, FINT*, FINT, double*, CINTOpt*, double*)
ISDF_DECLARE_2E(int2e_sph);
ISDF_DECLARE_2E(int2e_cart);
ISDF_DECLARE_2E(int2e_ip1_sph);
ISDF_DECLARE_2E(int2e_ip1_cart);
ISDF_DECLARE_2E(int2e_ip1ip2_sph);
ISDF_DECLARE_2E(int2e_ip1ip2_cart);
ISDF_DECLARE_2E(int2e_ipvip1_sph);
ISDF_DECLARE_2E(int2e_ipvip1_cart);
ISDF_DECLARE_2E(int2e_ipvip1ipvip2_sph);
ISDF_DECLARE_2E(int2e_ipvip1ipvip2_cart);
#undef ISDF_DECLARE_2E
}

// Owns libcint's atom, basis and environment arrays. Input arrays must use
// libcint's ATM/BAS/ENV layout and contain valid offsets into env.
class LibcintWrapper {
public:
    enum class Integral { int2e, int2e_ip1, int2e_ip1ip2,
                          int2e_ipvip1, int2e_ipvip1ipvip2 };
    enum class Basis { spherical, cartesian };

    LibcintWrapper(std::vector<FINT> atm, std::vector<FINT> bas,
                   std::vector<double> env, FINT natm, FINT nbas)
        : atm_(std::move(atm)), bas_(std::move(bas)), env_(std::move(env)),
          natm_(natm), nbas_(nbas)
    {
        if (natm < 0 || nbas < 0 ||
            atm_.size() < static_cast<std::size_t>(natm) * ATM_SLOTS ||
            bas_.size() < static_cast<std::size_t>(nbas) * BAS_SLOTS ||
            env_.size() < PTR_ENV_START) {
            throw std::invalid_argument("invalid libcint ATM/BAS/ENV table sizes");
        }
    }

    LibcintWrapper(const LibcintWrapper&) = delete;
    LibcintWrapper& operator=(const LibcintWrapper&) = delete;
    LibcintWrapper(LibcintWrapper&&) noexcept = default;
    LibcintWrapper& operator=(LibcintWrapper&&) noexcept = default;

    FINT natm() const noexcept { return natm_; }
    FINT nbas() const noexcept { return nbas_; }
    FINT* atm_data() noexcept { return atm_.data(); }
    const FINT* atm_data() const noexcept { return atm_.data(); }
    FINT* bas_data() noexcept { return bas_.data(); }
    const FINT* bas_data() const noexcept { return bas_.data(); }
    double* env_data() noexcept { return env_.data(); }
    const double* env_data() const noexcept { return env_.data(); }
    const std::vector<FINT>& atm_vec() const noexcept { return atm_; }
    const std::vector<FINT>& bas_vec() const noexcept { return bas_; }
    const std::vector<double>& env_vec() const noexcept { return env_; }

    // For a shell quartet, returns dimensions in libcint's i,j,k,l order.
    std::array<FINT, 4> dimensions(const std::array<FINT, 4>& shells,
                                   Basis basis = Basis::spherical) const
    {
        check_shells(shells);
        std::array<FINT, 4> dims{};
        for (std::size_t i = 0; i < 4; ++i) {
            dims[i] = basis == Basis::spherical
                ? CINTcgto_spheric(shells[i], bas_.data())
                : CINTcgto_cart(shells[i], bas_.data());
        }
        return dims;
    }

    static int components(Integral kind)
    {
        switch (kind) {
        case Integral::int2e: return 1;
        case Integral::int2e_ip1: return 3;
        case Integral::int2e_ip1ip2:
        case Integral::int2e_ipvip1: return 9;
        case Integral::int2e_ipvip1ipvip2: return 81;
        }
        throw std::invalid_argument("unknown libcint integral");
    }

    // out must contain components(kind)*product(dims) doubles. Pass out=nullptr
    // to query the required cache size (in doubles); cache=nullptr lets libcint
    // allocate temporary storage. CINTOpt, if supplied, must match `kind`.
    CACHE_SIZE_T evaluate(Integral kind, double* out,
                          const std::array<FINT, 4>& dims,
                          const std::array<FINT, 4>& shells,
                          Basis basis = Basis::spherical,
                          CINTOpt* optimizer = nullptr,
                          double* cache = nullptr) const
    {
        check_shells(shells);
        const auto minimum = dimensions(shells, basis);
        for (std::size_t i = 0; i < 4; ++i) {
            if (dims[i] < minimum[i]) {
                throw std::invalid_argument("integral dimension is smaller than shell size");
            }
        }
        auto mutable_dims = dims;
        auto mutable_shells = shells;
        return select(kind, basis)(out, mutable_dims.data(), mutable_shells.data(),
                                   const_cast<FINT*>(atm_.data()), natm_,
                                   const_cast<FINT*>(bas_.data()), nbas_,
                                   const_cast<double*>(env_.data()), optimizer, cache);
    }

    CACHE_SIZE_T cache_size(Integral kind, const std::array<FINT, 4>& shells,
                            Basis basis = Basis::spherical) const
    {
        return evaluate(kind, nullptr, dimensions(shells, basis), shells, basis);
    }

    // Convenience wrappers for each requested integral; defaults to spherical.
    CACHE_SIZE_T int2e(double* out, const std::array<FINT, 4>& dims,
                       const std::array<FINT, 4>& shells, Basis basis = Basis::spherical) const
    { return evaluate(Integral::int2e, out, dims, shells, basis); }

    CACHE_SIZE_T int2e_ip1(double* out, const std::array<FINT, 4>& dims,
                           const std::array<FINT, 4>& shells, Basis basis = Basis::spherical) const
    { return evaluate(Integral::int2e_ip1, out, dims, shells, basis); }

    CACHE_SIZE_T int2e_ip1ip2(double* out, const std::array<FINT, 4>& dims,
                              const std::array<FINT, 4>& shells, Basis basis = Basis::spherical) const
    { return evaluate(Integral::int2e_ip1ip2, out, dims, shells, basis); }

    CACHE_SIZE_T int2e_ipvip1(double* out, const std::array<FINT, 4>& dims,
                              const std::array<FINT, 4>& shells, Basis basis = Basis::spherical) const
    { return evaluate(Integral::int2e_ipvip1, out, dims, shells, basis); }

    CACHE_SIZE_T int2e_ipvip1ipvip2(double* out, const std::array<FINT, 4>& dims,
                                    const std::array<FINT, 4>& shells, Basis basis = Basis::spherical) const
    { return evaluate(Integral::int2e_ipvip1ipvip2, out, dims, shells, basis); }

private:
    // Internal function pointer is defined here rather than relying on the
    // optional CINTIntegralFunction typedef in libcint's generated header.
    using Function = decltype(&::int2e_sph);

    static Function select(Integral kind, Basis basis)
    {
        switch (kind) {
        case Integral::int2e: return basis == Basis::spherical ? ::int2e_sph : ::int2e_cart;
        case Integral::int2e_ip1: return basis == Basis::spherical ? ::int2e_ip1_sph : ::int2e_ip1_cart;
        case Integral::int2e_ip1ip2: return basis == Basis::spherical ? ::int2e_ip1ip2_sph : ::int2e_ip1ip2_cart;
        case Integral::int2e_ipvip1: return basis == Basis::spherical ? ::int2e_ipvip1_sph : ::int2e_ipvip1_cart;
        case Integral::int2e_ipvip1ipvip2: return basis == Basis::spherical ? ::int2e_ipvip1ipvip2_sph : ::int2e_ipvip1ipvip2_cart;
        }
        throw std::invalid_argument("unknown libcint integral");
    }

    void check_shells(const std::array<FINT, 4>& shells) const
    {
        for (FINT shell : shells) {
            if (shell < 0 || shell >= nbas_) {
                throw std::out_of_range("shell index is outside the basis table");
            }
        }
    }

    // libcint's C API takes non-const pointers for these input tables.
    std::vector<FINT> atm_;
    std::vector<FINT> bas_;
    std::vector<double> env_;
    FINT natm_;
    FINT nbas_;
};
