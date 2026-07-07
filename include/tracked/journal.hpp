#pragma once
// Per-thread JSONL log buffer for Tracked<T> instrumentation (schema v0.3).
// Namespace: tracked::journal  (not tracked::log, which is the math log function in ops.hpp)
//
// v0.3 schema change (breaking): every record now carries a stable per-value
// `id`, `in` holds the *direct-operand* ids verbatim (no primary_id heuristic),
// and the flat `prov` set is split into `prov_vars` (source-variable roots) and
// `prov_consts` (named constants).  See docs/PROVENANCE.md.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
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
    std::string              op;           // "add","sub","mul","div","neg","sqrt","exp","log","abs","opaque"
    std::string              at;           // "file:function:line", empty when not captured
    std::string              id;           // stable id of the produced value
    std::vector<std::string> in;           // direct-operand ids, verbatim
    double                   val;          // output value
    double                   cond;         // local condition number for this op
    double                   rel_err;      // accumulated relative error bound on the output
    std::vector<std::string> prov_vars;    // source-variable roots feeding this value
    std::vector<std::string> prov_consts;  // named constants feeding this value
};

// ---- Journal (thread-local log buffer) --------------------------------------

namespace journal {

namespace detail {
    inline thread_local std::vector<LogRecord> buf;

    // ---- Callsite counters (own the id-generation state so clear() can reset
    // them; make_id() in tracked.hpp reads/writes this map) --------------------
    struct CallsiteKey {
        std::string_view file;
        int              line;
        std::string_view op;
        bool operator<(const CallsiteKey& o) const {
            return std::tie(file, line, op) < std::tie(o.file, o.line, o.op);
        }
    };
    inline thread_local std::map<CallsiteKey, std::uint64_t> callsite_counters;
    inline thread_local std::uint64_t                        anon_counter = 0;

    // ---- Lazily-built lookup caches over `buf` -------------------------------
    inline thread_local std::unordered_map<std::string, std::size_t> id_index;
    inline thread_local std::set<std::string>                        source_names;
    inline thread_local bool                                         caches_dirty = true;

    inline void rebuild_caches() {
        id_index.clear();
        source_names.clear();
        for (std::size_t i = 0; i < buf.size(); ++i) {
            id_index[buf[i].id] = i;
            source_names.insert(buf[i].prov_vars.begin(), buf[i].prov_vars.end());
        }
        caches_dirty = false;
    }

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

inline void clear() {
    detail::buf.clear();
    detail::callsite_counters.clear();
    detail::anon_counter = 0;
    detail::id_index.clear();
    detail::source_names.clear();
    detail::caches_dirty = true;
}

inline void emit(std::string_view op,
                 const SourceLocation& loc,
                 std::string id,
                 std::vector<std::string> in_ids,
                 double val,
                 double cond,
                 double rel_err,
                 const std::set<std::string>& prov_vars,
                 const std::set<std::string>& prov_consts) {
    detail::buf.push_back({
        std::string(op),
        detail::format_loc(loc),
        std::move(id),
        std::move(in_ids),
        val, cond, rel_err,
        std::vector<std::string>(prov_vars.begin(),   prov_vars.end()),
        std::vector<std::string>(prov_consts.begin(), prov_consts.end())
    });
    detail::caches_dirty = true;
}

inline void flush(const std::string& path) {
    std::ofstream out(path);
    for (const auto& r : detail::buf) {
        out << "{\"op\":\""         << r.op  << '"'
            << ",\"at\":\""         << r.at  << '"'
            << ",\"id\":\""         << r.id  << '"'
            << ",\"in\":"           << detail::str_array(r.in)
            << ",\"val\":"          << detail::double_to_json(r.val)
            << ",\"cond\":"         << detail::double_to_json(r.cond)
            << ",\"rel_err\":"      << detail::double_to_json(r.rel_err)
            << ",\"prov_vars\":"    << detail::str_array(r.prov_vars)
            << ",\"prov_consts\":"  << detail::str_array(r.prov_consts);
        out << "}\n";
    }
}

// ---- Graph walk helpers -----------------------------------------------------
//
// The journal is a DAG: each record's `id` is a node, and `in` lists the ids of
// the operands that produced it.  The library owns the traversal so consumers
// don't reinvent it.  All three are O(depth) after the id index is built.

// id -> record-index map over the current buffer, rebuilt lazily after any emit.
inline const std::unordered_map<std::string, std::size_t>& index() {
    if (detail::caches_dirty) detail::rebuild_caches();
    return detail::id_index;
}

// Direct operands of `id`. Empty for source variables / constants / unknown ids
// (anything that never appears as the produced value of an op record).
inline std::vector<std::string> parents(std::string_view id) {
    const auto& idx = index();
    auto it = idx.find(std::string(id));
    if (it == idx.end()) return {};
    return detail::buf[it->second].in;
}

// Source-variable roots that ultimately fed `id`.  BFS backward through `in`
// edges; a visited id is a source iff it was introduced by track() (i.e. it
// appears in some record's prov_vars).  Constants and opaque markers are
// excluded — they never appear in prov_vars.
inline std::set<std::string> trace_sources(std::string_view id) {
    index();  // ensure caches (source_names) are current
    std::set<std::string> sources;
    std::set<std::string> visited;
    std::queue<std::string> q;
    q.push(std::string(id));
    while (!q.empty()) {
        std::string cur = std::move(q.front());
        q.pop();
        if (!visited.insert(cur).second) continue;
        if (detail::source_names.count(cur)) sources.insert(cur);
        for (auto& p : parents(cur))
            if (!visited.count(p)) q.push(p);
    }
    return sources;
}

// All causal ancestors of `id` (including `id`) in topological order: every id
// appears after all of its parents (roots first, `id` last).
inline std::vector<std::string> trace_ancestors(std::string_view id) {
    index();
    std::vector<std::string> order;
    std::set<std::string>    visited;
    std::function<void(const std::string&)> dfs = [&](const std::string& n) {
        if (!visited.insert(n).second) return;
        for (auto& p : parents(n)) dfs(p);
        order.push_back(n);
    };
    dfs(std::string(id));
    return order;
}

} // namespace journal
} // namespace tracked
