#pragma once
// Math function overloads for Tracked<T>: sqrt, exp, log, abs.
// Condition numbers are closed-form per-op values from numerical analysis.
//
// Found via ADL: write sqrt(x), exp(x), etc. for x of type Tracked<T>.
// For source location capture, pass TRACKED_HERE explicitly:
//   auto y = tracked::sqrt(x, TRACKED_HERE);

#include <tracked/tracked.hpp>
#include <cmath>

namespace tracked {

// sqrt(x): cond = 0.5 (relative error in output is half that of input)
template <class T>
Tracked<T> sqrt(const Tracked<T>& a, SourceLocation loc = {}) {
    using std::sqrt;
    T res      = sqrt(a.value_);
    T cond     = T(0.5);
    T new_err  = cond * (a.rel_err_bound_ + unit_roundoff<T>());
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("sqrt", loc, {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

// exp(x): cond = |x|  (large |x| amplifies input error)
template <class T>
Tracked<T> exp(const Tracked<T>& a, SourceLocation loc = {}) {
    using std::exp;
    T res      = exp(a.value_);
    T cond     = std::abs(a.value_);
    T new_err  = cond * (a.rel_err_bound_ + unit_roundoff<T>());
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("exp", loc, {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

// log(x): cond = 1/|log(x)|  (blows up near x=1 where log(x)→0)
template <class T>
Tracked<T> log(const Tracked<T>& a, SourceLocation loc = {}) {
    using std::log;
    T res     = log(a.value_);
    T ln_abs  = std::abs(res);
    // Cap condition at 1/u when |log(x)| is smaller than u (x ≈ 1).
    T cond    = (ln_abs > unit_roundoff<T>()) ? T(1) / ln_abs : T(1) / unit_roundoff<T>();
    T new_err  = cond * (a.rel_err_bound_ + unit_roundoff<T>());
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("log", loc, {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

// abs(x): cond = 1  (no amplification)
template <class T>
Tracked<T> abs(const Tracked<T>& a, SourceLocation loc = {}) {
    using std::abs;
    T res      = abs(a.value_);
    T cond     = T(1);
    T new_err  = cond * (a.rel_err_bound_ + unit_roundoff<T>());
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("abs", loc, {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

// sin(x): cond = |x·cos(x)/sin(x)| = |x·cot(x)|
// Blows up near integer multiples of π. Capped at 1/u when |sin(x)| < u·|x|.
template <class T>
Tracked<T> sin(const Tracked<T>& a, SourceLocation loc = {}) {
    using std::sin; using std::cos; using std::abs;
    T res    = sin(a.value_);
    T u      = unit_roundoff<T>();
    T abs_x  = abs(a.value_);
    T abs_s  = abs(res);
    T cond   = (abs_s >= u * abs_x)
             ? abs_x * abs(cos(a.value_)) / abs_s
             : T(1) / u;
    T new_err  = cond * (a.rel_err_bound_ + u);
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("sin", loc, {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

// cos(x): cond = |x·sin(x)/cos(x)| = |x·tan(x)|
// Blows up near π/2 + kπ. Capped at 1/u when |cos(x)| < u·|x|.
template <class T>
Tracked<T> cos(const Tracked<T>& a, SourceLocation loc = {}) {
    using std::sin; using std::cos; using std::abs;
    T res    = cos(a.value_);
    T u      = unit_roundoff<T>();
    T abs_x  = abs(a.value_);
    T abs_c  = abs(res);
    T cond   = (abs_c >= u * abs_x)
             ? abs_x * abs(sin(a.value_)) / abs_c
             : T(1) / u;
    T new_err  = cond * (a.rel_err_bound_ + u);
    T new_cond = std::max(a.max_cond_seen_, cond);
    auto prov  = a.provenance_;
    journal::emit("cos", loc, {detail::primary_id(a.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

// atan2(y, x): cond = 2·|x·y| / ((x²+y²)·|atan2(y,x)|)
// Pathological near atan2→0 (positive real axis) and at origin.
// Capped at 1/u when |atan2(y,x)| < u.
template <class T>
Tracked<T> atan2(const Tracked<T>& y, const Tracked<T>& x, SourceLocation loc = {}) {
    using std::atan2; using std::abs;
    T res      = atan2(y.value_, x.value_);
    T u        = unit_roundoff<T>();
    T abs_res  = abs(res);
    T denom    = (x.value_ * x.value_ + y.value_ * y.value_) * abs_res;
    T numer    = T(2) * abs(x.value_ * y.value_);
    T cond     = (abs_res >= u) ? numer / denom : T(1) / u;
    T max_in_err = std::max(y.rel_err_bound_, x.rel_err_bound_);
    T new_err    = cond * (max_in_err + u);
    auto prov    = detail::prov_union(y.provenance_, x.provenance_);
    T new_cond   = std::max({y.max_cond_seen_, x.max_cond_seen_, cond});
    journal::emit("atan2", loc,
        {detail::primary_id(y.provenance_), detail::primary_id(x.provenance_)},
        (double)res, (double)cond, (double)new_err, prov);
    return Tracked<T>(res, new_err, new_cond, std::move(prov));
}

} // namespace tracked
