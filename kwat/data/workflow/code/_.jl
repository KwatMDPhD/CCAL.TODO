using Revise

using BenchmarkTools
using PyCall

using Kwat

kwat = pyimport("kwat")

se = joinpath("..", "input", "setting.json")

PAR, PAI, PAC, PAO = Kwat.workflow.get_path(se)
