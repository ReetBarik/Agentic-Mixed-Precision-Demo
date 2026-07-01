#pragma once
// Tracked<T> — floating-point wrapper that propagates condition numbers,
// accumulated relative error bounds, and provenance alongside the value,
// and emits a structured JSONL log of every op.
//
// Usage:
//   #include <tracked/tracked.hpp>
//   #include <tracked/ops.hpp>     // for sqrt, exp, log, abs
//
//   auto a = tracked::track("a", 1.0);
//   auto b = tracked::track("b", 1e8);
//   auto r = a - b;                          // operator, no source location
//   auto r = sub(a, b, TRACKED_HERE);        // named fn, captures file:func:line
//
//   tracked::journal::flush("out.jsonl");

#include <tracked/journal.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <string_view>

namespace tracked {

// ---- Numeric constants ------------------------------------------------------

template <class T>
constexpr T unit_roundoff() {
    return std::numeric_limits<T>::epsilon() / T(2);
}

// ---- Internal helpers -------------------------------------------------------

namespace detail {

inline std::set<std::string> prov_union(const std::set<std::string>& a,
                                         const std::set<std::string>& b) {
    std::set<std::string> r = a;
    r.insert(b.begin(), b.end());
    return r;
}

inline std::string primary_id(const std::set<std::string>& p) {
    return p.empty() ? "_" : *p.begin();
}

} // namespace detail

// ---- Forward declarations (definitions after the class) ---------------------

template <class T> class Tracked;
template <class T> Tracked<T> add(const Tracked<T>&, const Tracked<T>&, SourceLocation loc = {});
template <class T> Tracked<T> sub(const Tracked<T>&, const Tracked<T>&, SourceLocation loc = {});
template <class T> Tracked<T> mul(const Tracked<T>&, const Tracked<T>&, SourceLocation loc = {});
template <class T> Tracked<T> div(const Tracked<T>&, const Tracked<T>&, SourceLocation loc = {});
template <class T> Tracked<T> neg(const Tracked<T>&, SourceLocation loc = {});

template <class T, class... Inputs>
Tracked<T> opaque_at(std::string_view fn_name, T value, SourceLocation loc,
                     const Inputs&... inputs);

// ---- Tracked<T> -------------------------------------------------------------

template <class T>
class Tracked {
public:
    T                     value_;
    T                     rel_err_bound_;  // accumulated relative error bound
    T                     max_cond_seen_;  // max local condition on any op producing this
    std::set<std::string> provenance_;     // contributing source-variable ids

    Tracked() : value_(T(0)), rel_err_bound_(T(0)), max_cond_seen_(T(0)) {}

    explicit Tracked(T v)
        : value_(v), rel_err_bound_(unit_roundoff<T>()), max_cond_seen_(T(0)) {}

    Tracked(T v, T err, T cond, std::set<std::string> prov)
        : value_(v), rel_err_bound_(err), max_cond_seen_(cond)
        , provenance_(std::move(prov)) {}

    T value()    const { return value_; }
    T rel_err()  const { return rel_err_bound_; }
    T max_cond() const { return max_cond_seen_; }
    const std::set<std::string>& provenance() const { return provenance_; }

    // Operators — call named free functions with empty source location.
    Tracked operator+(const Tracked& b) const { return tracked::add(*this, b); }
    Tracked operator-(const Tracked& b) const { return tracked::sub(*this, b); }
    Tracked operator*(const Tracked& b) const { return tracked::mul(*this, b); }
    Tracked operator/(const Tracked& b) const { return tracked::div(*this, b); }
    Tracked operator-()                 const { return tracked::neg(*this); }

    Tracked& operator+=(const Tracked& b) { *this = *this + b; return *this; }
    Tracked& operator-=(const Tracked& b) { *this = *this - b; return *this; }
    Tracked& operator*=(const Tracked& b) { *this = *this * b; return *this; }
    Tracked& operator/=(const Tracked& b) { *this = *this / b; return *this; }

    // Comparisons return plain bool — no Tracked propagation.
    bool operator==(const Tracked& b) const { return value_ == b.value_; }
    bool operator!=(const Tracked& b) const { return value_ != b.value_; }
    bool operator< (const Tracked& b) const { return value_ <  b.value_; }
    bool operator> (const Tracked& b) const { return value_ >  b.value_; }
    bool operator<=(const Tracked& b) const { return value_ <= b.value_; }
    bool operator>=(const Tracked& b) const { return value_ >= b.value_; }
};

// ---- Factory functions ------------------------------------------------------

// Tag a fresh input with a source-variable id.
template <class T>
Tracked<T> track(std::string id, T value) {
    Tracked<T> r(value);
    r.provenance_ = {std::move(id)};
    return r;
}

// Opaque-call barrier: value crosses a non-Tracked boundary.
//
// Semantics: pass-through.  cond = 1 (we have no insight into what the opaque
// function does, so the conservative null hypothesis is that input errors flow
// through unchanged).  rel_err = max(input_rel_errs) + u (one extra roundoff
// for the boundary crossing).  Provenance = union(input_provs) U {fn_name}
// (preserve the upstream attribution chain, add a marker that the value
// crossed an opaque boundary).
//
// Two forms:
//   opaque(fn_name, value)                  -- no tracked inputs (e.g., raw
//                                              return from an external call
//                                              with no tracked args).  cond=1,
//                                              rel_err=u, prov={fn_name}.
//   opaque(fn_name, value, in1, in2, ...)   -- preserves error + provenance
//                                              from the tracked inputs that
//                                              were stripped to .value() before
//                                              the external call.

namespace detail {

// Fold tracked inputs into (max_rel_err, union_prov).  Empty pack => (0, {}).
template <class T>
inline void opaque_fold(T& /*max_err*/, std::set<std::string>& /*prov*/) {}

template <class T, class... Rest>
inline void opaque_fold(T& max_err, std::set<std::string>& prov,
                        const Tracked<T>& head, const Rest&... rest) {
    if (head.rel_err_bound_ > max_err) max_err = head.rel_err_bound_;
    prov.insert(head.provenance_.begin(), head.provenance_.end());
    opaque_fold(max_err, prov, rest...);
}

} // namespace detail

template <class T, class... Inputs>
Tracked<T> opaque(std::string_view fn_name, T value,
                  const Inputs&... inputs) {
    return opaque_at<T, Inputs...>(fn_name, value, SourceLocation{}, inputs...);
}

// Same, but with an explicit source location.  Use this form when calling
// from a wrapper that wants TRACKED_HERE attribution.
template <class T, class... Inputs>
Tracked<T> opaque_at(std::string_view fn_name, T value, SourceLocation loc,
                     const Inputs&... inputs) {
    T max_in_err = T(0);
    std::set<std::string> prov;
    detail::opaque_fold(max_in_err, prov, inputs...);
    prov.insert(std::string(fn_name));

    T cond    = T(1);
    T new_err = cond * (max_in_err + unit_roundoff<T>());

    journal::emit("opaque", loc,
        {std::string(fn_name)},
        (double)value, (double)cond, (double)new_err, prov);

    return Tracked<T>(value, new_err, cond, std::move(prov));
}

// ---- Named free functions (M1–M3): include source location ------------------
//
// Operators above delegate to these with loc={} (no location).
// For location capture, call directly:
//   auto r = sub(a, b, TRACKED_HERE);
//
// Error bound: new_err = local_cond * (max(input_rel_errs) + unit_roundoff<T>())
// This is the standard first-order model: fl(a op b) = (a op b)(1 + delta), |delta| <= u.

template <class T>
Tracked<T> add(const Tracked<T>& a, const Tracked<T>& b, SourceLocation loc) {
    T res     = a.value_ + b.value_;
    T abs_res = std::abs(res);
    T abs_sum = std::abs(a.value_) + std::abs(b.value_);
    T cond    = (abs_res > T(0)) ? abs_sum / abs_res
              : (abs_sum == T(0)) ? T(1) : T(1) / unit_roundoff<T>();
    T max_in_err = std::max(a.rel_err_bound_, b.rel_err_bound_);
    T new_err    = cond * (max_in_err + unit_roundoff<T>());
    T new_cond   = std::max({a.max_cond_seen_, b.max_cond_seen_, cond});
    auto prov    = detail::prov_union(a.provenance_, b.provenance_);
    journal::emit("add", loc,
        {detail::primary_id(a.provenance_), detail::primary_id(b.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

template <class T>
Tracked<T> sub(const Tracked<T>& a, const Tracked<T>& b, SourceLocation loc) {
    T res     = a.value_ - b.value_;
    T abs_res = std::abs(res);
    T abs_sum = std::abs(a.value_) + std::abs(b.value_);
    T cond    = (abs_res > T(0)) ? abs_sum / abs_res
              : (abs_sum == T(0)) ? T(1) : T(1) / unit_roundoff<T>();
    T max_in_err = std::max(a.rel_err_bound_, b.rel_err_bound_);
    T new_err    = cond * (max_in_err + unit_roundoff<T>());
    T new_cond   = std::max({a.max_cond_seen_, b.max_cond_seen_, cond});
    auto prov    = detail::prov_union(a.provenance_, b.provenance_);
    journal::emit("sub", loc,
        {detail::primary_id(a.provenance_), detail::primary_id(b.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

template <class T>
Tracked<T> mul(const Tracked<T>& a, const Tracked<T>& b, SourceLocation loc) {
    T res        = a.value_ * b.value_;
    T cond       = T(1);
    T max_in_err = std::max(a.rel_err_bound_, b.rel_err_bound_);
    T new_err    = cond * (max_in_err + unit_roundoff<T>());
    T new_cond   = std::max({a.max_cond_seen_, b.max_cond_seen_, cond});
    auto prov    = detail::prov_union(a.provenance_, b.provenance_);
    journal::emit("mul", loc,
        {detail::primary_id(a.provenance_), detail::primary_id(b.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

template <class T>
Tracked<T> div(const Tracked<T>& a, const Tracked<T>& b, SourceLocation loc) {
    T res        = a.value_ / b.value_;
    T cond       = T(1);
    T max_in_err = std::max(a.rel_err_bound_, b.rel_err_bound_);
    T new_err    = cond * (max_in_err + unit_roundoff<T>());
    T new_cond   = std::max({a.max_cond_seen_, b.max_cond_seen_, cond});
    auto prov    = detail::prov_union(a.provenance_, b.provenance_);
    journal::emit("div", loc,
        {detail::primary_id(a.provenance_), detail::primary_id(b.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

template <class T>
Tracked<T> neg(const Tracked<T>& a, SourceLocation loc) {
    T res      = -a.value_;
    T cond     = T(1);
    T new_err  = cond * (a.rel_err_bound_ + unit_roundoff<T>());
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("neg", loc,
        {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

} // namespace tracked
