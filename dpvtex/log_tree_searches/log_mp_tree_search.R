reinstall_phangorn <- function() {
  # install phangorn from branch log-mp-search-trees in lenacoll's fork of the repo
  print("(re)install phangorn")
  if (isNamespaceLoaded("package:phangorn")) detach("package:phangorn", unload = TRUE)
  if ("phangorn" %in% installed.packages()[, "Package"]) {
    remove.packages("phangorn")
  }
  remotes::install_github("lenacoll/phangorn@log-mp-search-trees")
  library(phangorn)
}

log_mp_search_list <- function(msa, output_log, seed = NULL, start_tree_type = "random") {
  # perform mp trees search with optim.parsimony
  # start_tree_type: "random" for random tree (rtree), "nj" for neighbor joining
  type <- NULL
  if (endsWith(msa, ".fasta") || endsWith(msa, ".fa")) {
    type <- "fasta"
  } else if (endsWith(msa, ".nex") || endsWith(msa, ".nexus")) {
    type <- "nexus"
  } else {
    stop("Unsupported file format. Please provide a .fasta or .nex file.")
  }
  phy_data <- read.phyDat(msa, format = type)

  # If seed is provided, set it for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
    print(paste("Using seed:", seed))
  }

  if (start_tree_type == "random") {
    # random starting tree using rtree
    seq_names <- names(phy_data)
    start_tree <- rtree(length(seq_names), tip.label = seq_names)
    start_tree$edge.length <- NULL
    print("Using random start tree (rtree)")
  } else {
    # use neighbor joining tree as start tree
    print("Using neighbor joining start tree")
    start_tree <- NJ(dist.hamming(phy_data))
    start_tree$edge.length <- NULL
  }
  mp_tree <- optim.parsimony(tree = start_tree, data = phy_data, log = output_log)
}


main <- function() {
  # get input fasta and output logfile from command line
  library(optparse)
  option_list <- list(
    make_option(c("-f", "--msa"), type = "character", help = "Input msa file"),
    make_option(c("-o", "--output"), type = "character", help = "Output tree log file"),
    make_option(c("--seed"), type = "integer", default = NULL, help = "Random seed for reproducibility"),
    make_option(c("--start-tree"), type = "character", default = "random",
                help = "Start tree type: 'random' (rtree) or 'nj' (neighbor joining) [default: %default]")
  )
  opt_parser <- OptionParser(option_list = option_list)
  opt <- parse_args(opt_parser)


  # install phangorn from remote, if it isn't yet
  if (("phangorn" %in% installed.packages()[, "Package"])) {
    desc <- packageDescription("phangorn")
    github_info <- list(
      repo = desc$GithubRepo,
      username = desc$GithubUsername,
      branch = desc$GithubRef,
      sha = desc$GithubSHA1
    )
  } else {
    github_info <- list(repo = NULL)
  }
  if (is.null(github_info$repo) || github_info$username != "lenacoll" || github_info$branch != "log-mp-search-trees") reinstall_phangorn()
  library(phangorn) # load phangorn if it is already installed

  # perform tree search and log trees
  log_mp_search_list(msa = opt$msa, output_log = opt$output, seed = opt$seed,
                     start_tree_type = opt$"start-tree")
}

# Only run main when script is executed
if (!interactive()) {
  main()
}
