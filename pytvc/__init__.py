import pytvc.control
import pytvc.core
import pytvc.data
import pytvc.physics
from pytvc.physics import Vec3, Quat
import pathlib

cPath = f"{pathlib.Path(__file__).parent.resolve()}"

H13=f"{cPath}/motors/AeroTech_H13ST.rse"
F10=f"{cPath}/motors/Apogee_F10.rse"
C6=f"{cPath}/motors/Estes_C6.rse"
D12=f"{cPath}/motors/Estes_D12.rse"
E12=f"{cPath}/motors/Estes_E12.rse"
F15=f"{cPath}/motors/Estes_F15.rse"
