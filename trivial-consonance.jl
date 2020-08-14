using MIDI: pitch_to_name, name_to_pitch
using Plots
using ProgressMeter

# equal temperament ratio for the semitone distance
const κ = 2^(1/12)

# fundamental of the first note on a piano
const basefrequency = 27.5

# note -> frequency
notetofreq(n::Int) = κ^(n-1) * basefrequency

# frequency -> note (with a myopic rounding)
freqtonote(f::Float64)::Int = round(Int, log(κ, f / 27.5)) + 1

# midi number (21-108) ~ piano number (1-88)
toname(piano_number::Int) = pitch_to_name(20 + piano_number)

# the number of possible values from A0 (27.5 Hz) to the 5th harmonic of C8, E10 (20930 Hz)
const ℓ = name_to_pitch("E10") - 20

# the learning rate
const η = 0.015

# ■ training a single layer
input   = zeros(ℓ)
weights = zeros(ℓ, ℓ)

for i = 1:ℓ
    weights[i, i] = 1
end

@showprogress "Training... " for episode = 1:1e4
    for idx = 1:ℓ
        input[idx] = 0.0
    end

    # random note from the piano
    f = notetofreq(rand(1:88))
    harmonics = freqtonote.([1f, 2f, 3f, 4f, 5f])

    amplitude = 1.0

    # setting input's notes
    for idx in 1:length(harmonics)
        input[harmonics[idx]] = amplitude / idx
    end

    activation = weights * input

    for idx in 1:ℓ
        # Hebbian update
        weights[idx, :] .+= η * input * activation[idx]
    end
end

# ■ averaging log synaptic weights across activations from all piano notes
averaged = zeros(13)

for root = 1:88
    intervals = log.(weights[root, root:root+12])

    # upscaling to positive numbers
    intervals .-= minimum(intervals) - 0.25

    # geometric mean is the same as an arithmetic in this case
    global averaged = (averaged * (root - 1) + intervals) / root
end

# ■ plot
interval_names = ["Root", "m2", "M2", "m3", "M3", "P4", "Tritone", "P5", "m6", "M6", "m7", "M7", "Octave"]

bar(
    averaged,
    title="Consonance of two-tone intervals",
    dpi=150, size=(600, 400),
    ylabel="log synaptic weights (arbitrary units)",
    xticks=([1:length(interval_names);], interval_names),
    color=:darkviolet,
    linecolor=nothing,
    legend=:none,
)

figname = "reproduce-ranking.png"
savefig(figname)
println("open $figname to view the plot!")
