using Serialization, Printf
a = deserialize("moving_branch.jls"); b = deserialize("moving_parent.jls")
for k in sort(collect(keys(a))); endswith(k, "dts") && continue
    u, v = a[k], b[k]; @printf("%-36s max|branch - parent|/max|u| = %.3e\n", k, maximum(abs.(u .- v)) / maximum(abs, v))
end
