using Distributions
using AugmentedGaussianProcesses
using LinearAlgebra
using Random: seed!
using ValueHistories
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
N_data = 1000
N_test = 100
N_dim = 1
noise = 1e-16
minx=-1.0
maxx=1.0

l= sqrt(0.5); vf = 2.0;vg=10.0; α = 1.0;n_sig = 2
μ_0 = -5.0
kernel = RBFKernel(l,variance = vf); m = 40
kernel_g = RBFKernel(l,variance = vg)
autotuning = false

rmse(y,y_test) = norm(y-y_test,2)/sqrt(length(y_test))
logit(x) = 1.0./(1.0.+exp.(-x))
h(x) = α*logit.(x)
# h(x) = exp.(-x)
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_tot = vcat(X,x_test)
y_m = rand(MvNormal(zeros(N_data+N_test),kernelmatrix(X_tot,kernel)+1e-5*I))
y_noise = rand(MvNormal(μ_0*ones(N_data+N_test),kernelmatrix(X_tot,kernel_g)+1e-5I))
y = y_m .+ rand.(Normal.(0,h.(y_noise)))
scatter(X_tot,y_m)
scatter!(X_tot,y_noise)
display(scatter!(X_tot,y))
X_test = collect(x_test)
y_test = y[N_data+1:end]
y = y[1:N_data]
s = sortperm(vec(X))
miny = minimum(y); maxy = maximum(y)
miny = miny < 0 ? 1.5*miny : 0.5*miny
maxy = maxy > 0 ? 1.5*maxy : 0.5*maxy
# X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;

fullm = true
sparsem =false
stochm = false
println("Testing the Heteroscedastic model")
global metrics = MVHistory()
function callback(model,iter)
    push!(metrics,:λ,iter,median(model.λ))
    push!(metrics,:c,iter,median(model.c))
    push!(metrics,:γ,iter,median(model.γ))
    push!(metrics,:θ,iter,median(model.θ))
    push!(metrics,:μ_g,iter,median(abs.(model.μ_g)))
    pg = plot(X[s],y_noise[1:N_data][s],lab="true g")
    plot!(pg,X[s],model.μ_g[s],lab="μ_g")
    pl = plot(X[s],model.λ[s],title="λ")
    pθ = plot(X[s],model.θ[s],title="θ")
    pγ = plot(X[s],model.γ[s],title="γ")
    model.μ = copy(y_m[1:model.nSamples])
    display(plot(pg,pl,pθ,pγ))
    sleep(0.1)
end

# if fullm
println("Testing the full model")
t_full = @elapsed global model = AugmentedGaussianProcesses.BatchHGP(X,y,noise=noise,kernel=kernel,verbose=verbose,Autotuning=autotuning,α=α,kernel_g=kernel_g,μ_0=μ_0)
t_full += @elapsed model.train(iterations=10,callback=callback)
global y_full,sig_full = model.predictproba(X_test); rmse_full = rmse(y_full,y_test);
global y_fullg, sig_fullg = AugmentedGaussianProcesses.noisepredict(model,X_test)
if doPlots
    # p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="StudentT")
    p1=plot(x_test,y_full,lab="",title="Heteroscedastic",ylim=(miny,maxy))
    plot!(p1,X_test,y_full+n_sig*sqrt.(sig_full),fill=(y_full-n_sig*sqrt.(sig_full)),alpha=0.3,lab="Heteroscedastic GP")
    plot!(twinx(),X_test,y_fullg,lab="Latent g")

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
    y_nonoise = y_m[N_data+1:end]
    y_gnoise = y_noise[N_data+1:end]
    noise_ytest = h.(y_noise[N_data+1:end])
    ptrue=plot(X,y,t=:scatter,lab="Training points")
    plot!(ptrue,x_test,y_test,t=:scatter,lab="Test points",title="Truth")
    plot!(ptrue,x_test,y_nonoise,ylim=(miny,maxy),lab="Noiseless")
    plot!(ptrue,x_test,y_nonoise+n_sig*sqrt.(noise_ytest),fill=y_nonoise-n_sig*sqrt.(noise_ytest),alpha=0.3,lab="")
    plot!(twinx(),x_test,y_gnoise,lab="")
    # ptrue=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
    # plot!(ptrue,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
    display(plot(ptrue,ps...))
end


plot(X[s],y_noise[1:N_data][s],lab="true g")
plot!(X[s],model.μ_g[s],lab="μ_g")
# return true