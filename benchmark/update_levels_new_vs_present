using Plots, Chairmarks, DynamicDiscreteSamplers
function push_v!(ds, n)
    k = ds.track_info.nvalues
    for i in k+1:k+n
        push!(ds, i, 2.0^i)
    end
    return ds
end
function remove_v!(ds, n)
    k = ds.track_info.nvalues
    for i in k:-1:k-n+1
        delete!(ds, i)
    end
    return ds
end

x = 1:256;

y1 = [(@b DynamicDiscreteSampler() push_v!(_, xi) evals=1 seconds=.01).time for xi in x];
y2 = [(@b push_v!(DynamicDiscreteSampler(), xi) push_v!(_, xi) evals=1 seconds=.01).time for xi in x];
y3 = [(@b push_v!(DynamicDiscreteSampler(), xi) remove_v!(_, xi) evals=1 seconds=.01).time for xi in x];
y4 = [(@b push_v!(push_v!(DynamicDiscreteSampler(), xi), xi) remove_v!(_, xi) evals=1 seconds=.01).time for xi in x];

plot(x,y1,label="push! - new sampled levels");
plot!(x,y2,label="push! - present sampled levels");
plot!(x,y3,label="delete! - new sampled levels");
plot!(x,y4,label="delete! - present sampled levels")

savefig("update_levels_new_vs_present.png")
