using Documenter, SingularIntegrals

DocMeta.setdocmeta!(SingularIntegrals, :DocTestSetup, :(using SingularIntegrals, ClassicalOrthogonalPolynomials); recursive=true)

makedocs(;
    modules = [SingularIntegrals],
    sitename = "SingularIntegrals.jl",
    authors = "Sheehan Olver",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["Home" => "index.md"],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/JuliaApproximation/SingularIntegrals.jl.git",
    devbranch = "main",
    push_preview = true,
)
