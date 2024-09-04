import argparse
parser = argparse.ArgumentParser(description='ini')
parser.add_argument('config', type=str, nargs=1)
parser.add_argument('-sky', dest='Simulation pipeline to generate HILC component seperated CMB maps', action='store_true', help='map making')
parser.add_argument('-n', type=int, help='number of simulations')
args = parser.parse_args()

ini = args.inifile[0]

if args.sky:
    from echolens import simulation
    sim = simulation.CMBbharatSky.from_file(ini)
    sim.make_sims(args.n)