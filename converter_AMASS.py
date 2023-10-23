import argparse
from anim import amass
from anim import bvh
from anim.animation import Animation

parser = argparse.ArgumentParser()

parser.add_argument("file", type =str, default="data/moto_LO.bvh")
parser.add_argument("--static", action="store_true")
#parser.add_argument("--file", type =str, default="data/moto_LO.bvh")

args = parser.parse_args()

anim_bvh: Animation = bvh.load(filepath=args.file)

amass.save_as_npz(anim_bvh,"output3.npz",16,"female", static=args.static)