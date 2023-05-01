# Import relevant Packages

using LinearAlgebra
using Statistics
using Plots
using CSV
using DataFrames
using ColorSchemes
using MAT
import Dates
using ProgressMeter

# This package will send a notification to your desktop when code blocks are done running.
# It's a quality of life package for code that can take more than a few minutes to run.
using Alert


# Reverse function for sorting least-to-greatest
rev = x -> -x
# Format to print datetimes in, for saving files
dateformat = "yyyy-mm-dd-HH-MM"
# Path to exported CSVs
exportpath = "./exports/"
mkpath(exportpath)
# Coloblind safe color scheme, credits to https://jacksonlab.agronomy.wisc.edu/2016/05/23/15-level-colorblind-friendly-palette/
clrs = ["#000000","#004949","#009292","#ff6db6","#ffb6db", "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff", "#920000","#924900","#db6d00","#24ff24","#ffff6d"];

# Some names of norms
spectral = opnorm
frob = norm


###############################
## Impliment Core Algorithms ##
###############################


## Online Modified Gram Schmidt

function online_MGS!(Q, k, x, scratch, k0=1)
    # Q is the past result of Gram Schmidt -- an already orthonormal matrix
    # The first k-1 columns of Q are already full, so we should add to col k
    # x is the new column being added, will be edited in-place
    # This updates Q in-place and doesn't return anything
    # scratch is a vector of size n, the height of Q, and is to be overwritten arbitrarily
    # k0 is the column index to start orthogonalizing from, since Lanczos will not want us to reortho going all the way back to the first index
    if k == 1
        Q[:, 1] = x / norm(x)
    elseif k == 2
        Q_sub = view(Q, :, 1)
        y = x - (Q_sub' * x)*Q_sub
        Q[:, 2] = y / norm(y)
    else
        for i=k0:k-1
            # Efficient Memory management below.
            # Uses BLAS and in-place operations explicitly.
            copyto!(scratch, view(Q, :, i))
            u = x'*scratch
            rmul!(scratch, u)
            BLAS.axpy!(-1, scratch, x)
        end
        Q[:, k] = x / norm(x)
    end
end

## Single Vector Kyrlov with Full Orthogonalization
## Full orthogonalization implimented by online modified Gram Schmidt (OMGS)

function single_vec_krylov(A, k, iters)
    (n,d) = size(A)
    scratch = zeros(n) # scratch space for OMGS

    # Starting vector & Krylov matrix
    g = randn(n,1)
    K = zeros(n, iters+1)
    K[:,1] = g / norm(g)

    # Core loop: Compute matvecs and add to K via OMGS
    for t=1:iters
        x = A*(A'* K[:,t])
        online_MGS!(K, t+1, x, scratch)
    end

    # Pulling good low-rank approx from the span
    Z = Matrix(qr(K).Q)
    F = A' * Z
    M = F' * F
    M = Symmetric(M)
    Uk = eigen(M, sortby=rev).vectors[:,1:k]
    return Z * Uk
end

# A function that maps number of iterations to number of matvecs
single_vec_matvecs = iters -> 3*iters + 1
single_vec_subspace_size = iters -> iters + 1


## Block Kyrlov with Full Orthogonalization
## Full orthogonalization implimented by online modified Gram Schmidt (OMGS)

function block_krylov(A, k, b, iters)
    (n,d) = size(A)
    scratch = zeros(n) # scratch space for OMGS

    # Create initial block
    K = zeros(n, (iters+1)*b)
    for i=1:b
        online_MGS!(K, i, randn(n), scratch)
    end

    # Krylov Loop
    for t=1:iters
        # The block we're going to add to the span
        K_add = A* (A' * K[:,(t-1)*b+1:t*b])
        for i=1:b
            # Adding the columns one-by-one via OMGS
            online_MGS!(K, t*b + i, K_add[:,i], scratch)
        end
    end

    # Pulling good low-rank approx from the span
    Z = Matrix(qr(K).Q)
    M = Symmetric(Z' * A * A' * Z)
    Uk = eigen(M, sortby=rev).vectors[:,1:k]
    return Z * Uk
end

# A function that maps number of iterations to number of matvecs
block_matvecs = (iters, b) -> (3*iters+1)*b
block_subspace_size = (iters, b) -> (iters + 1)*b


## Compute relative errors in Norms

function relative_err(A, k, best_error, Q_est, norm_func)
    # A is the input matrix
    # k is the target rang
    # best_error should be norm_func(A_k) where A_k is the best rank-k approx to A
        # This can end up taking a lot of computation, so we precompute the best_error
    # norm_func is the norm we're low-rank approximating in
    est_error = norm_func(A - Q_est*Q_est'*A)
    return abs(est_error - best_error) / best_error
end



###################################
## Generating Synthetic Matrices ##
###################################



# Given a list of eigenvalues, make a matrix with those eigs
# Returns A = QDQ' where D is the eigvals and Q is a random orthogonal matrix
# Also returns the singular vectors Q in sorted order
function generate_from_eigs(eigvals)
    d = length(eigvals)
    Q = Matrix(qr(randn(d,d)).Q)
    eigvals = sort(eigvals, rev=true)
    return (A = Q * Diagonal(eigvals) * Q', Q_true = Q)
end

# Given a list of eigenvalues, make a diagonal matrix with those eigs
# Also returns the singular vectors Q in sorted order
function generate_from_eigs_diag(eigvals)
    d = length(eigvals)
    Q = Diagonal(I + zeros(d,d))
    eigvals = sort(eigvals, rev=true)
    return (A = Diagonal(eigvals), Q_true = Q)
end

## Lil' exmaple to check this is making the right eigs

vs = sort(randn(10).^2, rev=true)
norm(eigvals(generate_from_eigs(vs).A, sortby=rev) - vs)



#######################################
## Meta-code for Running Experiments ##
#######################################



## Generate data for SVK running on A for many iterations, error measured in a specified norm

function generate_trial_data(A, k, iters_min, iters_max, resolution, n_trials, best_err, norm_func; range_mode=:log10, svk_func=single_vec_krylov)
    # A is the input matrix
    # k is the target rank
    # iters_min is the minimum num of iterations to run for
    # iters_max is the maximum num of iterations to run for
    # resolution is how many choices for iteration complexity to try out
        # so resolution 8 for min_iters = 10 and max_iters = 10^8 means the code will run SVK
        # for 10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, and 10^8 iterations
    # best_err is norm_func(A_k) where A_k is the best low-rank approx to A
    # norm_func is the norm we're computing the loss of low-rank approx in
    # range_mode is either :log10 or :linear, to say if the range of iterations should be linear or logarithmic
    # svk_func is the single vector krylov implimentation to use. Used to swap between different orthogonalization schemes

    data = zeros(resolution, n_trials)

    # iters_range is the list of iteration complexities we will run
    iters_range = zeros(resolution)
    if range_mode === :log10
        iters_range = Int.(floor.(10 .^ (range(log10(iters_min), log10(iters_max), length=resolution))))
    elseif range_mode === :linear
        iters_range = Int.(floor.(range(iters_min, iters_max, length=resolution)))
    else
        @error "Invalid range_mode given: " * string(range_mode)
    end

    # Run the actual experiments
    for i=1:resolution
        iters = iters_range[i]
        for j=1:n_trials
            data[i,j] = relative_err(A, k, best_err, svk_func(A, k, iters), norm_func)
        end
    end

    # Return a bunch of metadata about how this experiment was run.
    # This metadata is useful to streamline automatically filling in plot data
    return (data=data, k=k, b=1, iters_range=iters_range,
            n_trials=n_trials, resolution=resolution,
            norm_func=norm_func, size=size(A),
            matvec_range = single_vec_matvecs.(iters_range),
            subspace_size_range = single_vec_subspace_size.(iters_range))

end


## Same as the above function, but for block krylov with block size b

function generate_trial_block_data(A, k, b, iters_min, iters_max, resolution, n_trials, best_err, norm_func; range_mode=:linear, krylov_func=block_krylov)
    # krylov_func is the block krylov implimentation to use. Used to swap between different orthogonalization schemes

    # iters_range is the list of iteration complexities we will run
    iters_range = zeros(resolution)
    if range_mode === :log10
        iters_range = unique(Int.(floor.(10 .^ (range(log10(iters_min), log10(iters_max), length=resolution)))))
    elseif range_mode === :linear
        iters_range = unique(Int.(floor.(range(iters_min, iters_max, length=resolution))))
    else
        @error "Invalid range_mode given: " * string(range_mode)
    end

    # Can be much shorter that the original value of resolution in some cases
    # because iters_max - iters_min might be less the resolution given to us
    resolution = length(iters_range)

    data = zeros(resolution, n_trials)

    # Run the actual experiments
    for i=1:resolution
        iters = iters_range[i]
        for j=1:n_trials
            data[i,j] = relative_err(A, k, best_err, krylov_func(A, k, b, iters), norm_func)
        end
    end

    # Return a bunch of metadata about how this experiment was run.
    # This metadata is useful to streamline automatically filling in plot data
    return (data=data, k=k, b=b, iters_range=iters_range,
            n_trials=n_trials, resolution=resolution,
            norm_func=norm_func, size=size(A),
            matvec_range = block_matvecs.(iters_range, b),
            subspace_size_range = block_subspace_size.(iters_range, b))

end



####################################################
## Plot 1: Slower convergence as gap size shrinks ##
####################################################



function generate_gap_data()
    # Parameters that define our sample matrix.
    d1 = 1000
    k1 = 10
    gaps = 10 .^ (range(log10(1), log10(1*1e-10), length=8))
    resolution = 10
    ntrials = 500
    iters_min = 25
    iters_max = 34

    output = []

    @alert @showprogress for i=1:length(gaps)
        gap = gaps[i]

        # Generate the spectrum, as a vector called eigs1
        alpha = 1.1
        power_func = i -> alpha^(1-i)
        eigs1 = power_func.(1:Int(floor(d1/2)))
        eigs1 = sort([eigs1; eigs1 / (1+gap)], rev=true)

        A1, Q1_true = generate_from_eigs_diag(eigs1);
        output = [output; (data=generate_trial_data(A1, k1, iters_min, iters_max, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=:linear), gap=gap)]
    end

    return output
end

gap_data = generate_gap_data();

## Plot the gap_data experiment

# Range of iterations the algorithm run for
m_range = gap_data[1].data.iters_range

# meds = medians; ups = 75th percentile; downs = 25th percentile
# These will be passed onto the CSV file, should you choose to export it
meds = zeros(length(m_range), length(gap_data))
ups = zeros(length(m_range), length(gap_data))
downs = zeros(length(m_range), length(gap_data))

# For each gap size, collect information about median loss and error
for (rel_errs, i) in [(gap_data[i].data.data, i) for i=1:length(gap_data)]
    median_errs = clamp.(median(rel_errs, dims=2), 1e-15, 1e7)
    lower_clamp = minimum(median_errs)
    # Clamp the quantiles between the minimum median error that ever occurs and 1e7
    bot_errs = clamp.([quantile(row, 0.25) for row in eachrow(rel_errs)], lower_clamp, 1e7)
    top_errs = clamp.([quantile(row, 0.75) for row in eachrow(rel_errs)], lower_clamp, 1e7)
    meds[:, i] = median_errs
    ups[:, i] = top_errs
    downs[:, i] = bot_errs
end

# List of all gap sizes, extracted from the output of generate_gap_data()
gaps = [gap_data[i].gap for i=1:length(gap_data)]
gap_export_df = DataFrame(gaps = gaps);

# Build the actual plot
plot(title="Gap Decay Plot " *
           "(d=" * string(gap_data[1].data.size[1]) * ", k=" * string(gap_data[1].data.k) * ", t=" * string(gap_data[1].data.n_trials) * ")",
    xaxis=("gap size", :log10),
    yaxis=("relative error", :log10),
    legend=:topright)

# For each number-of-iterations-to-run
for i=1:length(m_range)

    # Pick a color
    clr=clrs[i]

    # Plot some data
    plot!(gaps, meds[i,:], label=m_range[i], m=:hex, c=clr)
    plot!(gaps, ups[i,:], fillrange = downs[i,:], fillalpha=0.1, label=false, alpha = 0, c=clr)

    # Give yourself a name
    labl = string(m_range[i]) * "iters"

    # Use that name to write to a dataframe of exportable data
    gap_export_df[!, labl * "Median"] = meds[i,:]
    gap_export_df[!, labl * "Top"] = ups[i,:]
    gap_export_df[!, labl * "Bot"] = downs[i,:]
end
plot!()


## Uncomment this to save the plot to a CSV
# csvname = ("plot_gap_data_d" * string(gap_data[1].data.size[1]) * "_k" * string(gap_data[1].data.k)
#                              * "_r" * string(gap_data[1].data.resolution) * "_trials" * string(gap_data[1].data.n_trials)
#                              * "__" * Dates.format(Dates.now(), dateformat) * ".csv")
# CSV.write(exportpath * csvname, gap_export_df)




#################################################
## Plots 2&3: Random Perturbation or b=2 helps ##
#################################################




function generate_perturb_data()

    # Parameters that define our sample matrix.
    d1 = 1000
    k1 = 50
    gap = 0
    resolution = 20
    ntrials = 10

    min_iters = 150
    max_iters = 500

    # Should we take the x-axis to be linear or logarithmic?
    range_mode = :log10
    # range_mode = :linear

    output = []

    # Generate the spectrum defined in section 6.2
    d3 = d1 - k1
    alpha = 1.005
    power_func = i -> alpha^(1-i)
    eigs1 = power_func.(1:d3)
    eigs1 = sort([eigs1; eigs1[1:k1] / (1+gap)], rev=true)

    # Generate the matrix
    A1, Q1_true = generate_from_eigs_diag(eigs1)

    # Generate Gaussian and Diagonal perturbation matrices
    G = randn(d1, d1)
    G_diag = Diagonal(2*rand(d1).-1)

    # Setup a loading bar.
    p = Progress(8, 1)

    # Block size 1, unperturbed
    output = [output; (data=generate_trial_data(A1, k1, min_iters, max_iters, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="svk")]
    next!(p)

    # Block size 2, unperturbed
    output = [output; (data=generate_trial_block_data(A1, k1, 2, floor(min_iters/2), floor(max_iters/2), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="b2")]
    next!(p)

    # Block size 3, unperturbed
    output = [output; (data=generate_trial_block_data(A1, k1, 3, floor(min_iters/3), floor(max_iters/3), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="b3")]
    next!(p)

    # Block size k, unperturbed
    output = [output; (data=generate_trial_block_data(A1, k1, k1, floor(min_iters/k1), floor(max_iters/k1), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="bk")]
    next!(p)
    
    for i=6:2:16
        # Block size 1, perturbed by noise of scale ~ 10^{-i}
        output = [output; (data=generate_trial_data(A1 + 10.0^(-i)*G, k1, min_iters, max_iters, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="g1e-"*string(i))]
        output = [output; (data=generate_trial_data(A1 + 10.0^(-i)*G_diag, k1, min_iters, max_iters, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="d1e-"*string(i))]
        next!(p)
    end

    alert("Done!")

    return output
end

perturb_data = generate_perturb_data();

## Plotting the perturbation data


plot(title="Perturbations Plot " *
           "(d=" * string(perturb_data[1].data.size[1]) * ", k=" * string(perturb_data[1].data.k) * ", t=" * string(perturb_data[1].data.n_trials) * ")",
    xaxis=("matvecs"), 
    # yaxis=("relative error", :log10),  # Uncomment this for a logarithmic y-axis
    yaxis=("relative error"),  # Uncomment this for a linear y-axis
    legend=:topright
)

# resolution = maximum number of x-coordinates of a single series on our plot will have
resolution = length(perturb_data[1].data.iters_range)

# Build a dataframe with `resolution` many rows
perturb_export_df = DataFrame(row = 1:resolution);

# We'll use difference transparencies. The first 4 plots will be fully opaque, the rest will be 25% visible.
alphas = [1 1 1 1 0.25*ones(15)']
for (m_range, k_range, rel_errs, labl, clr, alpha, i) in [(perturb_data[i].data.matvec_range, perturb_data[i].data.subspace_size_range, perturb_data[i].data.data, perturb_data[i].type, ["red"; clrs][i], alphas[i], i) for i=1:length(perturb_data)]
    
    # Compute the plottable statistics
    median_errs = clamp.(median(rel_errs, dims=2), 1e-15, 1e7)[:]
    lower_clamp = minimum(median_errs)
    bot_errs = clamp.([quantile(row, 0.25) for row in eachrow(rel_errs)], lower_clamp, 1e7)[:]
    top_errs = clamp.([quantile(row, 0.75) for row in eachrow(rel_errs)], lower_clamp, 1e7)[:]

    # Plot the statistics
    plot!(m_range, median_errs, label=labl, c=clr, m=:hex, alpha=alpha)
    plot!(m_range, bot_errs, fillrange = top_errs, fillalpha=0.125, label=false, alpha = 0, c=clr)

    # res_here = resolution (number of x-coordinates) for this series. This can vary a lot.
    res_here = length(m_range)

    # Write the statistics data to a dataframe, padded with zeros at the end.
    perturb_export_df[!, labl * "Matvecs"] = [m_range; zeros(resolution - res_here)]
    perturb_export_df[!, labl * "SubspaceSize"] = [k_range; zeros(resolution - res_here)]
    perturb_export_df[!, labl * "Median"] = [median_errs; zeros(resolution - res_here)]
    perturb_export_df[!, labl * "Top"] = [bot_errs; zeros(resolution - res_here)]
    perturb_export_df[!, labl * "Bot"] = [top_errs; zeros(resolution - res_here)]
end
plot!(legend=:topright)

## Uncomment this to save the dataframe to a CSV
csvname = ("plot_perturb_data_d" * string(perturb_data[1].data.size[1]) * "_k" * string(perturb_data[1].data.k)
                             * "_r" * string(perturb_data[1].data.resolution) * "_trials" * string(perturb_data[1].data.n_trials)
                             * "__" * Dates.format(Dates.now(), dateformat) * ".csv")
CSV.write(exportpath * csvname, string.(perturb_export_df))






###################################################
## Plot 4: Many different block size comparisons ##
###################################################





function generate_block_comp_data(mode)
    # Change "mode" to run with whichever matrix you want to use for the grid of plots in Section 6.4

    d1 = 1000
    resolution = 20
    ntrials = 10

    k1, min_iters, max_iters, range_mode = 50, 50, 500, :log10

    # Declare variables to be set within the following if/else block:
    eigs1 = zeros(d1)
    name = ""

    if mode == 1
        # Matrix 1: Same as the last plot
        alpha = 1.005
        power_func = i -> alpha^(1-i)
        eigs1 = power_func.(1:d1-k1)
        eigs1 = sort([eigs1; eigs1[1:k1]], rev=true)
        name = "perturb-plot"
    
    elseif mode == 2       
        # Matrix 2: eps^{1/3} lower bound, but large k
        # This is a matrix where we never reach the exponential rate, so
        # being able to simulate larger block sizes ain't all that helpful
        flipped_wishart_spectrum = i -> 1-(i/d1)^2
        eigs1 = sort(flipped_wishart_spectrum.(1:d1), rev=true)
        name = "wishart-lb"

    elseif mode == 3
        # Matrix 3: Exponential Decay
        alpha = 1.001
        power_func = i -> alpha^(1-i)
        eigs1 = power_func.(1:d1)
        name = string(alpha) * "-exp-law"
    
    elseif mode == 4
        # Matrix 4: Exponential Decay
        alpha = 1.01
        power_func = i -> alpha^(1-i)
        eigs1 = power_func.(1:d1)
        name = string(alpha) * "-exp-law"

    elseif mode == 5
        # Matrix 5: Exponential Decay
        alpha = 1.1
        power_func = i -> alpha^(1-i)
        eigs1 = power_func.(1:d1)
        name = string(alpha) * "-exp-law"
    
    elseif mode == 6
        # Matrix 6: Polynomial Decay
        alpha = 1.5
        power_func = i -> i^-alpha
        eigs1 = power_func.(1:d1)
        name = string(alpha) * "-poly-law"

    elseif mode == 7
        # Matrix 7: Polynomial Decay
        alpha = 0.5
        power_func = i -> i^-alpha
        eigs1 = power_func.(1:d1)
        name = string(alpha) * "-poly-law"

    elseif mode == 8
        # Matrix 8: Polynomial Decay
        alpha = 0.1
        power_func = i -> i^-alpha
        eigs1 = power_func.(1:d1)
        name = string(alpha) * "-poly-law"
    
    elseif mode == 9
        # Matrix 9: Real World Matrix 1

        eigs1 = matread("nd3k_SVD.mat")["S"]["s"][:] ## Assuming you have nd3k_SVD.mat from SuiteSpare stored locally.
        name = "nd3k.mat"
        
    elseif mode == 10
        # Matrix 10: Real World Matrix 2

        eigs1 = matread("human_gene2_SVD.mat")["S"]["s"][:] ## Assuming you have nd3k_SVD.mat from SuiteSpare stored locally.
        name = "human_gene2_SVD.mat"

    elseif mode == 11
        # Matrix 11: Real World Matrix 2

        eigs1 = matread("appu_SVD.mat")["S"]["s"][:] ## Assuming you have nd3k_SVD.mat from SuiteSpare stored locally.
        name = "appu_SVD.mat"
        
    elseif mode == 12
        # Matrix 12: Real World Matrix 2

        eigs1 = matread("exdata_1_SVD.mat")["S"]["s"][:] ## Assuming you have nd3k_SVD.mat from SuiteSpare stored locally.
        name = "exdata_1_SVD.mat"
    end


    # We now generate a matrix from the list of eigenvalues we computed
    A1, Q1_true = generate_from_eigs_diag(eigs1)
    output = []

    # List of block sizes to use
    bs = [1 2 3 k1 k1+4]

    @showprogress for b=bs

        # For some reason I coded block size 1 to be a different generate_trial_data function
        if b == 1
            output = [output; (data=generate_trial_data(A1, k1, min_iters, max_iters, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="svk")]
        else
            output = [output; (data=generate_trial_block_data(A1, k1, b, max(floor(min_iters/b), 1), floor(max_iters/b), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="b="*string(b))]
        end

    end
    
    alert("Done!")

    return output, name, range_mode
end


## WARNING
## This next code block runs the full experiment for all 12 plots in plot_fig_6.tex

for i=1:12
    block_comp_data, block_comp_name, block_comp_xscale = generate_block_comp_data(i);

    ## Build the plot, similar code to the previous plot.

    plot(title= block_comp_name * " Plot " *
            "(d=" * string(block_comp_data[1].data.size[1]) * ", k=" * string(block_comp_data[1].data.k) * ", t=" * string(block_comp_data[1].data.n_trials) * ")",
        xaxis=("matvecs"),
        yaxis=("relative error", :log10),  # Uncomment this for a log y-axis
        # yaxis=("relative error"),  # Uncomment this for a linear y-axis
        legend=:topright
    )

    # Make a dataframe to let us export the data
    resolution = length(block_comp_data[1].data.iters_range)
    block_comp_export_df = DataFrame(row = 1:resolution);

    # Iterate over all of our series to plot
    for (m_range, k_range, rel_errs, labl, clr, i) in [(block_comp_data[i].data.matvec_range, block_comp_data[i].data.subspace_size_range, block_comp_data[i].data.data, block_comp_data[i].type, clrs[i+2], i) for i=1:length(block_comp_data)]

        # Calculate statistics
        median_errs = clamp.(median(rel_errs, dims=2), 1e-15, 1e7)[:]
        lower_clamp = minimum(median_errs)
        bot_errs = clamp.([quantile(row, 0.25) for row in eachrow(rel_errs)], lower_clamp, 1e7)[:]
        top_errs = clamp.([quantile(row, 0.75) for row in eachrow(rel_errs)], lower_clamp, 1e7)[:]

        # Plot the data
        plot!(m_range, median_errs, label=labl, c=clr, m=:hex)
        plot!(m_range, bot_errs, fillrange = top_errs, fillalpha=0.125, label=false, alpha = 0, c=clr)

        # Store the results, padding with zeros where needed.
        res_here = length(m_range)
        block_comp_export_df[!, labl * "Matvecs"] = [m_range; zeros(resolution - res_here)]
        block_comp_export_df[!, labl * "SubspaceSize"] = [k_range; zeros(resolution - res_here)]
        block_comp_export_df[!, labl * "Median"] = [median_errs; zeros(resolution - res_here)]
        block_comp_export_df[!, labl * "Top"] = [bot_errs; zeros(resolution - res_here)]
        block_comp_export_df[!, labl * "Bot"] = [top_errs; zeros(resolution - res_here)]
    end
    plot!(legend=:topright)

    ## Uncomment this to save the data to CSV
    csvname = ("plot_block_comp_data_n_" * block_comp_name * "_d" * string(block_comp_data[1].data.size[1]) * "_k" * string(block_comp_data[1].data.k)
                                * "_r" * string(block_comp_data[1].data.resolution) * "_trials" * string(block_comp_data[1].data.n_trials)
                                * "__" * Dates.format(Dates.now(), dateformat) * ".csv")
    CSV.write(exportpath * csvname, string.(block_comp_export_df))
    print(csvname)
end


#######################################
## Plot 5: Orthogonalization Matters ##
#######################################

# We impliment 2 more core functions here:
#  1. single vector w/ Lanczos
#  2. block krylov w/ Lanczos



function single_vec_krylov_lanczos(A, k, iters)
    # This code exactly mirros `single_vec_krylov`, which does full orthogonalization,
    # except for the call to online modified Gram Schmidt (`online_MGS!`), where it
    # only orthogonalizes against the last 2 columns
    (n,d) = size(A)
    scratch = zeros(n)

    g = randn(n,1)
    K = zeros(n, iters+1)
    K[:,1] = g / norm(g)
    for t=1:iters
        x = A*(A'* K[:,t])
        online_MGS!(K, t+1, x, scratch, max(1,t-1)) # only look back 2 cols
    end
    Z = Matrix(qr(K).Q)
    F = A' * Z
    M = F' * F
    M = Symmetric(M)
    Uk = eigen(M, sortby=rev).vectors[:,1:k]
    return Z * Uk
end


function block_krylov_lanczos(A, k, b, iters)
    # This matches `block_krylov`, except that it only orthogonalizes against
    # the last 2 blocks, which appears in the call to online modified Gram
    # Schmidt (`online_MGS!`)
    (n,d) = size(A)
    scratch = zeros(n)

    K = zeros(n, (iters+1)*b)
    for i=1:b
        online_MGS!(K, i, randn(n), scratch)
    end

    for t=1:iters
        K_add = A* (A' * K[:,(t-1)*b+1:t*b])
        for i=1:b
            # Orthogonalize, starting from col (t-2)b+1, or 2 blocks ago
            online_MGS!(K, t*b + i, K_add[:,i], scratch, max(1, (t-2)*b+1))
        end
    end

    Z = Matrix(qr(K).Q)
    M = Symmetric(Z' * A * A' * Z)
    Uk = eigen(M, sortby=rev).vectors[:,1:k]
    return Z * Uk
end


## Actual data generating code


function generate_ortho_data()
    d1 = 1000
    resolution = 20
    ntrials = 100

    # Build the spectrum with exponential decay
    k1, min_iters, max_iters, range_mode = 50, 50, 500, :log10
    alpha = 1.1
    power_func = i -> alpha^(1-i)
    eigs1 = power_func.(1:d1)
    name = string(alpha) * "-exp-law"

    A1, Q1_true = generate_from_eigs(eigs1)
    @show cond(A1)
    output = []

    p = Progress(8, 1)

    # Block size 1, Lanczos
    output = [output; (data=generate_trial_data(A1, k1, min_iters, max_iters, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode, svk_func=single_vec_krylov_lanczos), type="svk_lanczos")]
    next!(p)

    # Block size 2, Lanczos
    output = [output; (data=generate_trial_block_data(A1, k1, 2, ceil(min_iters/2), floor(max_iters/2), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode, krylov_func=block_krylov_lanczos), type="b2_lanczos")]
    next!(p)
    
    # Block size k, Lanczos
    output = [output; (data=generate_trial_block_data(A1, k1, k1, ceil(min_iters/k1), floor(max_iters/k1), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode, krylov_func=block_krylov_lanczos), type="bk_lanczos")]
    next!(p)
    
    # Block size k+4, Lanczos
    output = [output; (data=generate_trial_block_data(A1, k1, k1+4, ceil(min_iters/(k1+4)), floor(max_iters/(k1+4)), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode, krylov_func=block_krylov_lanczos), type="bk+4_lanczos")]
    next!(p)
    
    # Block size 1, full orthogonalization
    output = [output; (data=generate_trial_data(A1, k1, min_iters, max_iters, resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="svk_ortho")]
    next!(p)
    
    # Block size 2, full orthogonalization
    output = [output; (data=generate_trial_block_data(A1, k1, 2, ceil(min_iters/2), floor(max_iters/2), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="b2_ortho")]
    next!(p)
    
    # Block size k, full orthogonalization
    output = [output; (data=generate_trial_block_data(A1, k1, k1, ceil(min_iters/k1), floor(max_iters/k1), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="bk_ortho")]
    next!(p)
    
    # Block size k+4, full orthogonalization
    output = [output; (data=generate_trial_block_data(A1, k1, k1+4, ceil(min_iters/(k1+4)), floor(max_iters/(k1+4)), resolution, ntrials, norm(eigs1[k1+1:end]), frob, range_mode=range_mode), type="bk+4_ortho")]
    next!(p)

    alert("Done!")

    return output, name, range_mode
end

ortho_data, ortho_name, ortho_xscale = generate_ortho_data();

## Plot the information

plot(title= ortho_name * " Plot " *
           "(d=" * string(ortho_data[1].data.size[1]) * ", k=" * string(ortho_data[1].data.k) * ", t=" * string(ortho_data[1].data.n_trials) * ")",
    xaxis=("matvecs", ortho_xscale),
    yaxis=("relative error", :log10),  # Uncomment for log y-axis
    # yaxis=("relative error"),  # Uncomment for linear y-axis
    legend=:topright
)

# Build a dataframe to store the output
resolution = length(ortho_data[1].data.iters_range)
ortho_export_df = DataFrame(row = 1:resolution);

# Iterate over the series to plot
for (m_range, k_range, rel_errs, labl, clr, i) in [(ortho_data[i].data.matvec_range, ortho_data[i].data.subspace_size_range, ortho_data[i].data.data, ortho_data[i].type, clrs[i+2], i) for i=1:length(ortho_data)]

    # Compute the statistics
    median_errs = clamp.(median(rel_errs, dims=2), 1e-15, 1e7)[:]
    lower_clamp = minimum(median_errs)
    bot_errs = clamp.([quantile(row, 0.25) for row in eachrow(rel_errs)], lower_clamp, 1e7)[:]
    top_errs = clamp.([quantile(row, 0.75) for row in eachrow(rel_errs)], lower_clamp, 1e7)[:]

    # Plot the statistics
    plot!(m_range, median_errs, label=labl, c=clr, m=:hex)
    plot!(m_range, bot_errs, fillrange = top_errs, fillalpha=0.125, label=false, alpha = 0, c=clr)

    # Write the statistics to the dataframe, padded by zeros
    res_here = length(m_range)
    ortho_export_df[!, labl * "Matvecs"] = [m_range; zeros(resolution - res_here)]
    ortho_export_df[!, labl * "SubspaceSize"] = [k_range; zeros(resolution - res_here)]
    ortho_export_df[!, labl * "Median"] = [median_errs; zeros(resolution - res_here)]
    ortho_export_df[!, labl * "Top"] = [bot_errs; zeros(resolution - res_here)]
    ortho_export_df[!, labl * "Bot"] = [top_errs; zeros(resolution - res_here)]
end
plot!(legend=:topright)


## Uncomment this to save the data to CSV
csvname = ("plot_ortho_data_n_" * ortho_name * "_d" * string(ortho_data[1].data.size[1]) * "_k" * string(ortho_data[1].data.k)
                             * "_r" * string(ortho_data[1].data.resolution) * "_trials" * string(ortho_data[1].data.n_trials)
                             * "__" * Dates.format(Dates.now(), dateformat) * ".csv")
#  For some reason, this didn't want to write, so I casted all the numbers in it to strings. That worked. Weird.
CSV.write(exportpath * csvname, string.(ortho_export_df))
