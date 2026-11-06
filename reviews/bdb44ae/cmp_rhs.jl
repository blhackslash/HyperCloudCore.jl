using Serialization, Printf
a = deserialize("rhs_branch.jls"); b = deserialize("rhs_parent.jls")
for k in sort(collect(keys(a)))
    u, v = a[k], b[k]
    d = maximum(abs.(u .- v)); s = maximum(abs, v)
    @printf("%-40s max|branch - parent| / max|u| = %.3e  %s\n", k, d / s, u == v ? "(bitwise)" : "")
end
