
# ##
# ENV["JULIA_PYTHONCALL_EXE"] = "/usr/local/Caskroom/miniconda/base/envs/confgf/bin/python"
# using PythonCall
# @py import confgf
# ##
# @py import torch_geometric
# ##
# using DataFrames,CSV
# data = DataFrame(CSV.File("data/train/tox21_dense_train.csv"))
# ##

# ##
# # using PyCall


# @py import pickle
# load_pickle = pywith()
# @py def load_pickle(fpath):    
#       with open(fpath, "rb") as f:
#         data = pickle.load(f)
#     return data

1-1

##
using DataFrames,Pkg,Pickle,MolecularGraph
import Base.Iterators


sdf=sdfilereader("data/tox21.sdf")
sdfmol = sdftomol("/data/tox21.sdf")
precalculate!(sdfmol)
molsvg = drawsvg(sdfmol,300,300)
display("image/svg+xml",  molsvg)
##


# 	for mol in mols
# 	    precalculate!(mol)
# 	end
mols = [mol for mol in sdf]
mols[1]

mol = parse(SMILES, "o1ccc2c1cncn2")
molsvg = drawsvg(mols, 150, 150)
# Note: aromatic bond notation (e.g. dashed cycle path or circle in the ring) is not supported yet
display("image/svg+xml",  molsvg)

for m in mols[1:10]
    molsvg = drawsvg(m, 150, 150)
    display("image/svg+xml",  molsvg)
end
    
