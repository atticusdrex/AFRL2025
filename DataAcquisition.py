from util import *

# Function to compute a single laminar flame speed value 
def compute_flame_speed(phi_val, gas_mech, T0, p0, this_fuel, this_oxidizer):
    try:
        gas = cantera.Solution(gas_mech)
        gas.TP = T0, p0
        gas.set_equivalence_ratio(phi_val, fuel=this_fuel, oxidizer=this_oxidizer)
        
        flame = cantera.FreeFlame(gas, width=0.07)
        flame.set_initial_guess()  # use a built-in temperature and species profile guess

        # Setting somewhat aggressive refine criteria to get a converged result
        flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
        flame.solve(loglevel=0, auto=True)

        return phi_val, flame.velocity[0]  # m/s
    except:
        # Handling solver errors 
        print("Solver did not converge...")
        return phi_val, np.nan

def get_data_parallel(phi, gas_mech, T0):
    # Parameters
    this_fuel = 'C2H4'
    this_oxidizer = 'O2:1.0, N2:3.76'
    p0 = 101325.0  # Pa

    # Computing the flame speeds in parallel 
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_flame_speed, phi_val, gas_mech, T0, p0, this_fuel, this_oxidizer) for phi_val in phi]
        for future in tqdm(as_completed(futures), total=len(phi), desc="Computing flame speeds"):
            results.append(future.result())

    # Sort results by phi value to keep ordering consistent
    results.sort(key=lambda x: x[0])
    phi_sorted, V_sim = zip(*results)

    # returning results that did not return nans 
    return np.array(phi_sorted)[~np.isnan(V_sim)].reshape(-1, 1), np.array(V_sim)[~np.isnan(V_sim)]

if __name__ == "__main__":
    # Temperatures 
    temps = [450, 550]

    # Looping through and saving the data
    for temp in temps:
        # Initializing Data Dictionary
        d = {
            0:{
                'mech':'mechanisms/7sp_3step_C2H4_AFRL_FS.yaml',
                'n_points':128, 
                'min':0.6, 'max':1.4
            },
            1:{
                'mech':'mechanisms/7sp_3step_C2H4_USAFA.yaml',
                'n_points':64, 
                'min':0.6, 'max':1.4
            }, 
            2:{
                'mech':'mechanisms/c2h4_23sp_66st_zettervall.yaml',
                'n_points':32, 
                'min':0.6, 'max':1.4
            },
            3:{
                'mech':'mechanisms/c2h4_32sp_206st_lu.yaml',
                'n_points':16, 
                'min':0.6, 'max':1.4
            },
            4:{
                'mech':'mechanisms/USCMechII_111sp_784st.yaml',
                'n_points':8,
                'min':0.6, 'max':1.4
            }
        }

        # Looping through each chemical mechanism
        for level in d.keys():
            # Generating phi values 
            phi_min, phi_max = d[level]['min'], d[level]['max']
            phi = np.linspace(phi_min, phi_max, d[level]['n_points'])

            # Getting data for these phi values 
            X, Y = get_data_parallel(phi, d[level]['mech'], temp)

            # Concatenating X data with temperature as a feature
            X = np.hstack(
                (X, np.ones_like(X)*temp)
            )

            # Storing the data in the dictionary
            d[level]['X'], d[level]['Y'] = X, Y
        
        # Writing data to a binary pickle file
        with open("FlameSpeedData%d.pkl" % (temp), "wb") as outfile:
            pickle.dump(d, outfile)
