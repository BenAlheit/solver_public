print("start")
import pyvista as pv

print("import pv")
import numpy as np

print("import np")
import matplotlib.pyplot as plt

print("import plt")

stress_comp = 0
E_comp = 0

E_name = "Strain11__Strain12__Strain13__Strain21__Strain22__Strain23__Strain31__Strain32__Strain33"
S_name = "Stress11__Stress12__Stress13__Stress21__Stress22__Stress23__Stress31__Stress32__Stress33"


def plot(file_name, label):
    x = []
    y = []
    # file_name = f'viscoplasticity-{name}/viscoplasticity-{name}.pvd'
    print(file_name)
    reader = pv.get_reader(file_name)
    for i_time in range(len(reader.time_values) - 1):
        print(f"{i_time + 1} / {len(reader.time_values) - 1}")
        reader.set_active_time_point(i_time)
        mesh = reader.read()[0]
        stress = np.mean(mesh[S_name], axis=0)
        E = np.mean(mesh[E_name], axis=0)
        x.append(E[E_comp])
        y.append(stress[stress_comp])
    plt.plot(x, y, marker='s', mfc='none', label=label)


def plot_f(file_name, label):
    x = []
    y = []
    # file_name = f'viscoplasticity-{name}/viscoplasticity-{name}.pvd'
    print(file_name)
    reader = pv.get_reader(file_name)
    for i_time in range(len(reader.time_values) - 1):
        print(f"{i_time + 1} / {len(reader.time_values) - 1}")
        reader.set_active_time_point(i_time)
        mesh = reader.read()[0]
        stress = np.mean(mesh["Plastic_yield_surface"], axis=0)
        E = np.mean(mesh[E_name], axis=0)
        x.append(E[E_comp])
        y.append(stress)
    plt.plot(x, y, marker='s', mfc='none', label=label)


# plot("./cmake-build-debug-wsl/ale-elastoplastic-ut.pvd", "Voce")
# plot_f("./cmake-build-debug-wsl/ale-elastoplastic-ut.pvd", "f")

plot("../../cmake-build-debug-cerecam-computer/elastoplastic-cube/elastoplastic-cube-np-1.pvd", "Voce")
plot_f("../../cmake-build-debug-cerecam-computer/elastoplastic-cube/elastoplastic-cube-np-1.pvd", "f")

plt.xlabel('$E_{11}$')
plt.ylabel(r'$\tau_{11}$')
plt.grid()
plt.legend()

plt.show()
