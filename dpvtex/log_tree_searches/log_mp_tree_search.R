reinstall_phangorn <- function(){
  # install phangorn from branch log-mp-search-trees in lenacoll's fork of the repo
  print("(re)install phangorn")
  if (isNamespaceLoaded("package:phangorn")) detach("package:phangorn", unload = TRUE)
  if ("phangorn" %in% installed.packages()[,"Package"]){
    remove.packages("phangorn")
  }
  remotes::install_github("lenacoll/phangorn@log-mp-search-trees")
  library(phangorn)
}

log_mp_search_list <- function(msa, output_log){
  # perform mp trees search with optim.parsimony, using a neighbour joining tree as starting tree
  type <- NULL
  if (endsWith(msa, ".fasta") || endsWith(msa, ".fa")) {
    type <- "fasta"
  } else if (endsWith(msa, ".nex") || endsWith(msa, ".nexus")) {
    type <- "nexus"
  } else {
    stop("Unsupported file format. Please provide a .fasta or .nex file.")
  }
  phy_data <- read.phyDat(msa, format=type)

  cleaned_msa <- as.matrix(phy_data)
  cleaned_msa[cleaned_msa == "-" | cleaned_msa == "n" | cleaned_msa == "N"] <- NA
  cleaned_msa <- as.DNAbin(cleaned_msa)
  # Write to file
  fasta_output <- sub("\\.(fasta|fa|nex|nexus)$", ".fasta", msa)
  write.phyDat(phy_data, file=fasta_output, format="fasta")

  # creat NJ starting tree for MP tree search, then tree search
  a <- as.DNAbin(phy_data)
  d <- dist.dna(a, model="raw")
  start_tree <- nj(d)
  start_tree$edge.length <- NULL
  mp_tree <- optim.parsimony(tree = start_tree, data = phy_data, log = output_log)
}


main <- function() {
  # get input fasta and output logfile from command line
  library(optparse)
  option_list <- list(
    make_option(c("-f", "--msa"), type="character", help="Input msa file"),
    make_option(c("-o", "--output"), type="character", help="Output tree log file")
  )
  opt_parser <- OptionParser(option_list=option_list)
  opt <- parse_args(opt_parser)
  
  
  # install phangorn from remote, if it isn't yet
  if (("phangorn" %in% installed.packages()[,"Package"])){
    desc <- packageDescription("phangorn")
    github_info <- list(
      repo = desc$GithubRepo,
      username = desc$GithubUsername,
      branch = desc$GithubRef,
      sha = desc$GithubSHA1
    )
  } else{
    github_info <- list(repo = NULL)
  }
  if (is.null(github_info$repo) || github_info$username != "lenacoll" || github_info$branch != "log-mp-search-trees") reinstall_phangorn()
  library(phangorn) # load phangorn if it is already installed
  
  # perform tree search and log trees
  log_mp_search_list(msa = opt$msa, output_log = opt$output)
}

# Only run main when script is executed
if (!interactive()) {
  main()
}
