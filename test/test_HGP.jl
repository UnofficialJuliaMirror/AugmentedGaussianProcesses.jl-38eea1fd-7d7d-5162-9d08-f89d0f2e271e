using Distributions
using AugmentedGaussianProcesses
using LinearAlgebra
using Random: seed!
seed!(42)
if !@isdefined doPlots
    doPlots = true
end
if !@isdefined verbose
    verbose = 3
end
if doPlots
    using Plots
    pyplot()
end
N_data = 100
N_test = 100
N_dim = 1
noise = 0.2
minx=-1.0
maxx=1.0
function latent(x)
    return x[:,1].*sin.(x[:,2])
end
l= sqrt(0.5); v = 2.0; α = 0.5
kernel = RBFKernel(l,variance = v); m = 40
kernel_g = RBFKernel(l*0.5,variance = v)
autotuning = false

rmse(y,y_test) = norm(y-y_test,2)/sqrt(length(y_test))
logit(x) = 1.0./(1.0.+exp.(-x))

X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_tot = vcat(X,x_test)
y_m = rand(MvNormal(zeros(N_data+N_test),kernelmatrix(X_tot,kernel)+1e-3*I))
y_noise = rand(MvNormal(zeros(N_data+N_test),kernelmatrix(X_tot,kernel_g)+1e-3I))
y = y_m .+ α*rand.(Normal.(0,logit.(y_noise)))
scatter(X_tot,y_m)
scatter!(X_tot,y_noise)
display(scatter!(X_tot,y))
X_test = collect(x_test)
y_test = y[N_data+1:end]
y = y[1:N_data]
miny = minimum(y); maxy = maximum(y)
# X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;

fullm = true
sparsem =false
stochm = false
println("Testing the StudentT model")

# if fullm
println("Testing the full model")
t_full = @elapsed global fullmodel = AugmentedGaussianProcesses.BatchHGP(X,y,noise=noise,kernel=kernel,verbose=verbose,Autotuning=autotuning,α=α,kernel_g=kernel_g)
t_full += @elapsed fullmodel.train(iterations=100)
# _ =  fullmodel.predict(X_test)
global y_full = fullmodel.predict(X_test); rmse_full = rmse(y_full,y_test);
global y_fullg, y_fullcovg = fullmodel.predictproba(X_test)
if doPlots
    # p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="StudentT")
    p1=plot(x_test,y_full,t=:scatter,lab="",title="Heteroscedastic",ylim=(1.5*miny,1.5*maxy))
    push!(ps,p1)
end
# end

if sparsem
    println("Testing the sparse model")
    t_sparse = @elapsed global sparsemodel = AugmentedGaussianProcesses.SparseStudentT(X,y,Stochastic=false,Autotuning=autotuning,verbose=verbose,m=m,noise=noise,kernel=kernel,ν=ν)
    t_sparse += @elapsed sparsemodel.train(iterations=1000)
    _ =  sparsemodel.predict(X_test)
    y_sparse = sparsemodel.predict(X_test); rmse_sparse = norm(y_sparse-y_test,2)/sqrt(length(y_test))
    if doPlots
        p2=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="Sparse StudentT")
        plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p2)
    end
end

if stochm
    println("Testing the sparse stochastic model")
    t_stoch = @elapsed stochmodel = AugmentedGaussianProcesses.SparseStudentT(X,y,Stochastic=true,batchsize=20,Autotuning=autotuning,verbose=verbose,m=m,noise=noise,kernel=kernel,ν=ν)
    t_stoch += @elapsed stochmodel.train(iterations=1000)
    _ =  stochmodel.predict(X_test)
    y_stoch = stochmodel.predict(X_test); rmse_stoch = norm(y_stoch-y_test,2)/sqrt(length(y_test))
    if doPlots
        p3=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(minx*1.1,maxx*1.1),lab="",title="Stoch. Sparse StudentT")
        plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p3)
    end
end
t_full != 0 ? println("Full model : RMSE=$(rmse_full), time=$t_full") : nothing
t_sparse != 0 ? println("Sparse model : RMSE=$(rmse_sparse), time=$t_sparse") : nothing
t_stoch != 0 ? println("Stoch. Sparse model : RMSE=$(rmse_stoch), time=$t_stoch") : nothing

if doPlots
    ptrue=plot(x_test,y_test,t=:scatter,lab="training points",title="Truth")
    plot!(ptrue,x_test,y_m[N_data+1:end],ylim=(1.5*miny,1.5*maxy))
    # ptrue=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
    # plot!(ptrue,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
    display(plot(ptrue,ps...))
end

return true
