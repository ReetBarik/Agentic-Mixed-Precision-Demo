#pragma once
// Per-thread JSONL log buffer for Tracked<T> instrumentation.
// Namespace: tracked::journal  (not tracked::log, which is the math log function in ops.hpp)

#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tracked {

// ---- Source location --------------------------------------------------------

struct SourceLocation {
    std::string_view file;
    std::string_view function;
    int line = 0;
};

// Capture call-site location. Place at a call site or pass as a default arg.
#define TRACKED_HERE ::tracked::SourceLocation{__FILE__, __func__, __LINE__}

// ---- Log record (one per op) ------------------------------------------------

struct LogRecord {
    std::string              op;       // "add","sub","mul","div","neg","sqrt","exp","log","abs","opaque"
    std::string              at;       // "file:function:line", empty when not captured
    std::vector<std::string> in;       // primary provenance id per input operand ("_" if unnamed)
    double                   val;      // output value
    double                   cond;     // local condition number for this op
    double                   rel_err;  // accumulated relative error bound on the output
    std::vector<std::string> prov;     // union provenance of the result
};

// ---- Journal (thread-local log buffer) --------------------------------------

namespace journal {

namespace detail {
    inline thread_local std::vector<LogRecord> buf;

    inline std::string double_to_json(double v) {
        if (std::isnan(v)) return "null";
        if (std::isinf(v)) return v > 0 ? "1.7976931348623157e+308" : "-1.7976931348623157e+308";
        std::ostringstream ss;
        ss << std::setprecision(17) << v;
        return ss.str();
    }

    inline std::string str_array(const std::vector<std::string>& v) {
        std::string r = "[";
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) r += ',';
            r += '"'; r += v[i]; r += '"';
        }
        r += ']';
        return r;
    }

    inline std::string format_loc(const SourceLocation& loc) {
        if (loc.line == 0) return "";
        return std::string(loc.file) + ':' + std::string(loc.function) + ':'
               + std::to_string(loc.line);
    }
} // namespace detail

inline const std::vector<LogRecord>& records() { return detail::buf; }
inline void clear() { detail::buf.clear(); }

inline void emit(std::string_view op,
                 const SourceLocation& loc,
                 std::vector<std::string> in_ids,
                 double val,
                 double cond,
                 double rel_err,
                 const std::set<std::string>& prov) {
    detail::buf.push_back({
        std::string(op),
        detail::format_loc(loc),
        std::move(in_ids),
        val, cond, rel_err,
        std::vector<std::string>(prov.begin(), prov.end())
    });
}

inline void flush(const std::string& path) {
    std::ofstream out(path);
    for (const auto& r : detail::buf) {
        out << "{\"op\":\""     << r.op  << '"'
            << ",\"at\":\""     << r.at  << '"'
            << ",\"in\":"       << detail::str_array(r.in)
            << ",\"val\":"      << detail::double_to_json(r.val)
            << ",\"cond\":"     << detail::double_to_json(r.cond)
            << ",\"rel_err\":"  << detail::double_to_json(r.rel_err)
            << ",\"prov\":"     << detail::str_array(r.prov);
        out << "}\n";
    }
}

} // namespace journal
} // namespace tracked
