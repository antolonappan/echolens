import argparse
import toml
parser = argparse.ArgumentParser(description='ini')
parser.add_argument('config', type=str, nargs=1)
parser.add_argument('-s','--sky', dest='sky', action='store_true', help='Simulation pipeline to generate HILC component seperated CMB maps')
parser.add_argument('-f','--filter', dest='filter', action='store_true', help='C inverse filtering pipeline for CS maps')
parser.add_argument('-r','--recon', dest='recon', action='store_true', help='Reconstruction pipeline for CS maps')
parser.add_argument('-n','--nsims', type=int, help='number of simulations')
args = parser.parse_args()

ini = args.config[0]

if args.nsims is None:
    try:
        nsims = toml.load(ini)['simulation']['nsims']
    except KeyError:
        raise ValueError('nsims not found in ini file or not provided as an argument, please provide nsims')

if args.sky:
    from echolens import simulation
    sim = simulation.CMBbharatSky.from_file(ini)
    sim.make_sims(args.nsims)

if args.filter:
    raise NotImplementedError

if args.recon:
    raise NotImplementedError