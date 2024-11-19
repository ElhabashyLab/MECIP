# MECIP
## A command line tool for machine learning driven Evolutionary Coupling based Interface Prediction


### Abstract

*Interacting proteins are likely to co-evolve over time. Thus, the large-scale analysis of co-evolutionary data can give insight in protein-protein interactions (PPIs) without using experimental high-throughput methods that can be error-prone in circumstances like analysing transient interactions.
MECIP is a highly modular inference method to detect protein interactions as well as identify likely interfaces on an amino acid level. This is achieved by a multi-step machine learning (ML) approach, which can use a multitude of inputs, like evolutionary coupling (EC) data, residue solvent accessibility (RSA), structures or models of the interacting subunits, and more, to make accurate predictions. Different models assess the likelihood of a protein pair to interact as well as the exact interaction interface. The following docking step can validate or refute the predictions made. With this newly gained information, the tool can iteratively refine its predictions by applying the machine learning models again, looking at geometrical consistencies between predictions.
By combining an easy-to-use command line interface with highly modular input possibilities and a detailed parameter selection, MECIP yields high accuracy interaction and interface prediction even for large-scale applications like whole proteome screenings.*





## How to run

MECIP can be accessed and executed completely from the command line. All of the parameters have to be included in a text file, which will be the only necessary input. Thus, an example call of MECIP looks like this:
> \>python mecip/src/main.py params\_file.txt

All of the important parameters, thresholds and user inputs can be accessed and adapted in the **params\_file.txt**, which will be discussed in the following.


## Params_file 
The file is compromised of 11 sections, each containing parameters for different parts of the tool. All adjustable parameters are documented in detail. The first section contains all required inputs. All of these inputs have to be changed according to your dataset. It contains crucial inputs like the location of test and training datasets, the output location, or the type of run that should to be executed. The next section deals with the predictors. The type of predictor, hyperparameter grids, feature lists, and imputation methods can be adjusted here amongst other inputs. After that comes the cluster section. The clustering algorithms have many parameters that can be modified here. A section regarding the residue solvent accessibility follows, containing thresholds and parameters that only deal with the RSA input and its effect on the tool. The next section contains options that alter the way data is read in. Different checks and additional computations while reading data can be toggled on or off. After that follows a section that deals with the output files. Similar to the last section, different kinds of plots and tables can be toggled here, to determine what should be created as an output for the given run. Next, the definition for an interaction can be adjusted. This includes the distance of two residues that determines an interaction as well as the atom type, that is considered while calculating the distances. The confidence thresholds section deals with the threshold moving problem after each prediction based on probabilities. All options discussed in section \ref{interaction prediction} can be regulated here. The following segment manages all parameters that can be changed for the HADDOCK run. This includes the creation of restraint and pdb files. Second to last comes the pdb section. It contains parameters and thresholds that determine which pdb file should be used as well as the details of the alignment to the UniProt sequence and the following modifications done. Lastly, all other parameters that do not belong to any other segment can be adjusted. This covers the number of iterations for both outer CVs, adding additional random resamplers to deal with imbalanced data, or the number of residue pairs that should be considered for the interaction prediction. The default configuration for this crucial selection of residue pairs is either the top $0.999$ percentile of all residue pairs included in the EC analysis, or the top 50 scoring pairs, whichever number is bigger. 

The parameter file included with the code can be used as a template for all parameters, only changing the ones that improve the performance given the unique characteristics of each dataset and biological application.# co-evolution_mitochondria
