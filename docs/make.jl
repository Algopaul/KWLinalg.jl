using Documenter, KWLinalg

push!(LOAD_PATH, "../src/")
makedocs(
 sitename="KWLinalg",
 pages = [
          "Home" => "index.md",
         ]
)
deploydocs(
  repo = "github.com/Algopaul/KWLinalg.jl.git",
  versions = nothing
)
