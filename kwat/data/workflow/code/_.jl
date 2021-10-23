using Revise
using BenchmarkTools
using PyCall

using Kwat

kwat = pyimport("kwat")

pas = joinpath("..", "input", "setting.json")

PAR, PAI, PAC, PAO = Kwat.workflow.get_path(pas)

SE = Kwat.workflow.read_setting(pas)

# ==============================================================================
