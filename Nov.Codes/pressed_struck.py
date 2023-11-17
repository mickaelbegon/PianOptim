"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, acos, dot, pi, Function
import time
import numpy as np
import biorbd_casadi as biorbd
import pickle
import os

from bioptim import (
    BiorbdModel,
    PenaltyController,
    ObjectiveList,
    PhaseTransitionFcn,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    CostType,
    PhaseTransitionList,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    MultinodeObjectiveList,
    Axis,
)

#
# def minimize_difference(all_pn: PenaltyNode):
#     return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx
#


# Joint indices in the biomechanical model:
# 0| .Pelvic Tilt, Anterior (-) and Posterior (+) Rotation
# 1| . Thorax, Left (+) and Right (-) Rotation
# 2| . Thorax, Flexion (-) and Extension (+)
# 3|0. Right Shoulder, Abduction (-) and Adduction (+)
# 4|1. Right Shoulder, Internal (+) and External (-) Rotation
# 5|2. Right Shoulder, Flexion (+) and Extension (-)
# 6|3. Elbow, Flexion (+) and Extension (-)
# 7|4. Elbow, Pronation (+) and Supination (-)
# 8|5. Wrist, Flexion (-) and Extension (+)
# 9|6. MCP, Flexion (+) and Extension (-)

# Note: The signs (+/-) indicate the direction of the movement for each joint.

# Description of movement phases:
# Phase 0: Preparation - Getting the fingers in position.
# Phase 1: Key Descend - The downward motion of the fingers pressing the keys.
# Phase 2: Key Bed - The phase where the keys are fully pressed and meet the bottom.
# Phase 3: Key Release (Upward) - Releasing the keys and moving the hand upward.
# Phase 4: Return to Neutral (Downward) - Bringing the fingers back to a neutral position, ready for the next action.


def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx


def custom_func_track_finger_5_on_the_right_of_principal_finger(controller: PenaltyController) -> MX:
    finger_marker_idx = biorbd.marker_index(controller.model.model, "finger_marker")
    markers = controller.mx_to_cx("markers", controller.model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    finger_marker_5_idx = biorbd.marker_index(controller.model.model, "finger_marker_5")
    markers_5 = controller.mx_to_cx("markers_5", controller.model.markers, controller.states["q"])
    finger_marker_5 = markers_5[:, finger_marker_5_idx]

    markers_diff_key2 = finger_marker[1] - finger_marker_5[1]

    return markers_diff_key2


def custom_func_track_principal_finger_and_finger5_above_bed_key(controller: PenaltyController, marker: str) -> MX:
    biorbd_model = controller.model
    finger_marker_idx = biorbd.marker_index(biorbd_model.model, marker)
    markers = controller.mx_to_cx("markers", biorbd_model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    markers_diff_key3 = finger_marker[2] - (0.07808863830566405 - 0.02)

    return markers_diff_key3


def custom_func_track_principal_finger_pi_in_two_global_axis(controller: PenaltyController, segment: str) -> MX:
    rotation_matrix_index = biorbd.segment_index(controller.model.model, segment)
    q = controller.states["q"].mx
    # global JCS gives the local matrix according to the global matrix
    principal_finger_axis = controller.model.model.globalJCS(q, rotation_matrix_index).to_mx()  # x finger = y global
    y = MX.zeros(4)
    y[:4] = np.array([0, 1, 0, 1])
    # @ x : pour avoir l'orientation du vecteur x du jcs local exprimé dans le global
    # @ produit matriciel
    principal_finger_y = principal_finger_axis @ y
    principal_finger_y = principal_finger_y[:3, :]

    global_y = MX.zeros(3)
    global_y[:3] = np.array([0, 1, 0])

    teta = acos(dot(principal_finger_y, global_y[:3]))
    output_casadi = controller.mx_to_cx("scal_prod", teta, controller.states["q"])

    return output_casadi


def prepare_ocp(allDOF, pressed, ode_solver) -> OptimalControlProgram:
    if allDOF:
        biorbd_model_path = "./Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"
        dof_wrist_finger = [8, 9]
        all_dof_but_wrist_finger = [0, 1, 2, 3, 4, 5, 6, 7]

    else:
        biorbd_model_path = "./Squeletum_hand_finger_3D_2_keys_octave_LA_without.bioMod"
        dof_wrist_finger = [5, 6]
        all_dof_but_wrist_finger = [0, 1, 2, 4]

    all_phases = [0, 1, 2, 3, 4]

    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    if pressed:
        # Velocity profile found thanks to the motion capture datas.
        vel_push_array = [0.0, -0.114, -0.181, -0.270, -0.347, -0.291, -0.100, ]
        n_shooting = (30, 7, 9, 10, 10)
        phase_time = (0.3, 0.044, 0.051, 0.15, 0.15)
    else:
        vel_push_array = [-0.698, -0.475, -0.368, -0.357, -0.368, -0.278, ]
        n_shooting = (30, 6, 9, 10, 10)
        phase_time = (0.3, 0.027, 0.058, 0.15, 0.15)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=4)

    # Objectives
    # Minimize Torques generated into articulations
    objective_functions = ObjectiveList()
    for phase in all_phases:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=1, index=all_dof_but_wrist_finger
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=1000, index=dof_wrist_finger
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=dof_wrist_finger #all_dof_but_wrist_finger
        )

    # for phase in [1, 2]:
    #     objective_functions.add(
    #         ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=phase, weight=0.0001, index=dof_wrist_finger
    #     )

    # Constraints
    constraints = ConstraintList()
    if pressed:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=0, node=Node.ALL,
            first_marker="finger_marker",
            second_marker="high_square",
        )
    else:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=0, node=Node.START,
            first_marker="finger_marker",
            second_marker="high_square",
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=0, node=Node.END,
            first_marker="finger_marker",
            second_marker="high_square",
        )

    #Stricly constrained velocity at start
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=1, node=Node.START,
        marker_index=4, axes=None if pressed else Axis.Z, #none means all
        target=vel_push_array[0]
    )

    for node in range(1, len(vel_push_array)): #todo: use Node.INTERMEDIATES
        constraints.add(
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            phase=1, node=node,
            marker_index=4, axes=Axis.Z,
            min_bound=-0.01, max_bound=0.01,
            target=vel_push_array[node],
        )

    # No finger's tip velocity at the end of phase 1
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=1, node=Node.END,
        marker_index=4,
    )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=1, node=Node.END,
        first_marker="finger_marker",
        second_marker="low_square",
    )



    ForceProfile = [30, 26, 24, 20, 16, 12, 8, 4, 0]
    for node in range(n_shooting[2]):
        for idx in [0, 1]:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                phase=2, node=node,
                contact_index=idx,
                min_bound=-ForceProfile[node] / 3, max_bound=ForceProfile[node] / 3,
                quadratic=False,
            )

        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            phase=2, node=node,
            contact_index=2,
            target=ForceProfile[node],
            quadratic=False,
            # min_bound=-0.1, max_bound=0.1,
        )


    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=3, node=Node.END,
        first_marker="MCP_contact_finger",
        second_marker="phase_3_upward",
    )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=4, node=Node.END,
        first_marker="finger_marker",
        second_marker="high_square",
    )


    for phase in all_phases:
        # To keep the index and the small finger above the bed key.
        constraints.add(
            custom_func_track_principal_finger_and_finger5_above_bed_key,
            phase=phase, node=Node.ALL,
            marker="finger_marker",
            min_bound=0,
            max_bound=np.inf,
        )
        constraints.add(
            custom_func_track_principal_finger_and_finger5_above_bed_key,
            phase=phase, node=Node.ALL,
            marker="finger_marker_5",
            min_bound=0,
            max_bound=np.inf,
        )
        # To keep the small finger on the right of the principal finger.
        constraints.add(
            custom_func_track_finger_5_on_the_right_of_principal_finger,
            phase=phase, node=Node.ALL,
            min_bound=0.00001,
            max_bound=np.inf,
        )
        # To keep the hand/index perpendicular of the key piano all long the attack.
        constraints.add(
            custom_func_track_principal_finger_pi_in_two_global_axis,
            phase=phase, node=Node.ALL,
            segment="secondmc",
            target=np.full((1, n_shooting[phase] + 1), pi / 2),
            min_bound=-np.pi / 24,
            max_bound=np.pi / 24,
            quadratic=False,
        )
        constraints.add(
            custom_func_track_principal_finger_pi_in_two_global_axis,
            phase=phase, node=Node.ALL,
            segment="2proxph_2mcp_flexion",
            target=np.full((1, n_shooting[phase] + 1), pi / 2),
            min_bound=-np.pi / 24,
            max_bound=np.pi / 24,
            quadratic=False,
        )
        # To block ulna rotation before the key pressing.
        constraints.add(
            ConstraintFcn.TRACK_STATE,
            phase=phase, node=Node.ALL,
            key="qdot",
            index=all_dof_but_wrist_finger[-1],  # prosupination
            min_bound=-1, max_bound=1,
            quadratic=False,
        )

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # States: bounds and Initial guess
    x_init = InitialGuessList()
    x_bounds = BoundsList()
    for phase in all_phases:
        x_bounds.add("q", bounds=biorbd_model[phase].bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=biorbd_model[phase].bounds_from_ranges("qdot"), phase=phase)

        x_init.add("q", [0] * biorbd_model[phase].nb_q, phase=phase)
        x_init.add("qdot", [0] * biorbd_model[phase].nb_q, phase=phase)

        # x_init[phase]["q"][4, 0] = 0.08 #todo put names
        # x_init[phase]["q"][5, 0] = 0.67
        # x_init[phase]["q"][6, 0] = 1.11
        # x_init[phase]["q"][7, 0] = 1.48
        # x_init[phase]["q"][9, 0] = 0.17

    if allDOF:
        x_bounds[0]["q"][[0], 0] = -0.1  # todo put names
        x_bounds[0]["q"][[2], 0] = 0.1

        x_bounds[4]["q"][[0], 2] = -0.1
        x_bounds[4]["q"][[2], 2] = 0.1

    # Define control path constraint and initial guess
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    for phase in all_phases:
        u_bounds.add(
            "tau",
            min_bound=[tau_min] * biorbd_model[phase].nb_tau,
            max_bound=[tau_max] * biorbd_model[phase].nb_tau,
            phase=phase,
        )
        u_init.add("tau", [tau_init] * biorbd_model[phase].nb_tau, phase=phase)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transition,
        ode_solver=ode_solver,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    print(os.getcwd())
    polynomial_degree = 4
    allDOF = False
    pressed = True #False means Struck
    dirName = "/Users/mickaelbegon/Library/CloudStorage/Dropbox/1_EN_COURS/FALL2023/"
    if allDOF:
        saveName = dirName + ("Pressed" if pressed else "Struck") + "_with_Thorax.pckl"
        nq = 10
    else:
        saveName = dirName + ("Pressed" if pressed else "Struck") +  "_without_Thorax.pckl"
        nq = 7

    ocp = prepare_ocp(allDOF=allDOF, pressed=pressed, ode_solver=OdeSolver.COLLOCATION(polynomial_degree=polynomial_degree))

    ocp.add_plot_penalty(CostType.ALL)

    # # --- Solve the program --- # #
    solv = Solver.IPOPT(show_online_optim=False)
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)

    # # --- Download datas on a .pckl file --- #
    q_sym = MX.sym("q_sym", nq, 1)
    qdot_sym = MX.sym("qdot_sym", nq, 1)
    tau_sym = MX.sym("tau_sym", nq, 1)

    phase = 2
    Contact_Force = Function(
        "Contact_Force",
        [q_sym, qdot_sym, tau_sym],
        [ocp.nlp[phase].model.contact_forces_from_constrained_forward_dynamics(q_sym, qdot_sym, tau_sym)],
    )

    rows = 9
    cols = 3
    F = [[0] * cols for _ in range(rows)]

    for i in range(0, 9):
        idx = i * (polynomial_degree + 1)
        F[i] = Contact_Force(
            sol.states[phase]["q"][:, idx], sol.states[phase]["qdot"][:, idx], sol.controls[phase]["tau"][:, i]
        )
    F_array = np.array(F)


    data = dict(
        states=sol.states,
        states_no_intermediate=sol.states_no_intermediate,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=np.array(sol.cost)[0][0],
        # detailed_cost=sol.detailed_cost,
        # real_time_to_optimize=sol.real_time_to_optimize,
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp],
        phase_time=sol.phase_time,
        Time=sol.time,
        Force_Values=F_array,
    )

    with open(saveName, "wb") as file:
        pickle.dump(data, file)
    #
    # print("Tesults saved")
    # print("Temps de resolution : ", time.time() - tic, "s")
    #
    # sol.print_cost()
    # ocp.print(to_console=False, to_graph=False)
    # # sol.graphs(show_bounds=True)
    # sol.animate(show_floor=False, show_global_center_of_mass=False, show_segments_center_of_mass=False, show_global_ref_frame=True, show_local_ref_frame=False, show_markers=False, n_frames=250,)


if __name__ == "__main__":
    main()
