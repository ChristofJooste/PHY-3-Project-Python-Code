import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from datetime import datetime


    #####################################
    #   #   #   #   #   #   #   #   #   #
    #   #   FUNCTIONS & PROCEDURES  #   #
    #   #   #   #   #   #   #   #   #   #
    #####################################


def rand_zero_one(): #   Function to obtain a random value between 0 and 1 
    R = random.uniform(0,1,1)   #   Produces (1) random floating point number in the range [0,1)
    return R 



def make_grid(L, temp):   #   Creates an L x L Ising grid with random spin-up/ spin-down elements
    grid = [[1 for i in range(L)] for j in range(L)]    #   Creates a 2-d list of size L x L, populated with only ones (makes the entire matrix spin-up)

    if temp > 1.5:  #   Ensures no "clumping" of dipoles at low temperatures
        for i in range(L):  #   Row number
            for j in range(L):     #    Column number
                if rand_zero_one() <= 0.5:   #   Sets random matrix elements to spin-down elements (i.e. you end up with a randomized L x L ising matrix)
                    grid[i][j] = -1

    return grid 



def rand_row_col(L): #   Function to obtain random row or column in L x L Ising grid (for metropolis algorithm).
    R = random.randint(0, L, 1)     #   Produces (1) random integer in the range [0,L)
    return R



def prob(deltaE, temp):    #   Probability that a spin flip will be accepted if the resulting change in energy would be >0.
    
    if temp == 0:
        factor = 0  #   This prevents division by zero in the exponent of e^(deltaE / kT)
    else:
        factor = np.exp(-1*deltaE/temp)   #   Probability factor e^(deltaE / kT) which will determine if the spin flips or not

    if rand_zero_one() < factor:
        return 1     #   Spin should flip
    else:
        return 0    # Spin should remain the same



def spin_flip_Ediff(i, j):  #   Looks at dipole at position [i][j] and returns the energy difference flipping the spin would cause
                            #   (Assumes periodic boundary conditons)
       
    if i == 0:
        top = L - 1 #   If dipole is in the top row, its above neighbour will be in the final row
    else:
        top = i - 1 #   Else its above neighbour is just above
    
    if i == L - 1:
        bottom = 0 #   If dipole is in the bottom row, its below neighbour will be in the first row
    else:
        bottom = i + 1 #   Else its below neighbour is just below

    if j == 0:
        left = L - 1 #   If dipole is in the leftmost column, its left neighbour will be in the final column
    else:
        left = j - 1 #   Else its left neighbour is just to the left

    if j == L - 1:
        right = 0 #   If dipole is in the rightmost column, its right neighbour will be in the first column
    else:
        right = j + 1 #   Else its right neighbour is just to the right

    first = grid_list[int(i)][int(left)]
    second = grid_list[int(top)][int(j)]
    third = grid_list[int(i)][int(right)]  
    fourth = grid_list[int(bottom)][int(j)]   #   Spins of the first, second, third, and fourth neighbour of the i-jth dipole (CW from the left)

    deltaE = 2*grid_list[i][j]*(first+second+third+fourth)

    return deltaE



def metropolis_algorithm(grid_list, L, temp):    #   Takes L x L grid, applies metropolis algorithm, produces represantitive sample at given temp

    N = L**2

    for flips in range(100*N):  #   Ensures each dipole gets a chance to flip ~100 times
        I = int(rand_row_col(L))
        J = int(rand_row_col(L))

        deltaE = spin_flip_Ediff(I, J)

        if deltaE <= 0:
            grid_list[I][J] = -1*grid_list[I][J]    #   If Ediff <= 0, flip I,Jth dipole
        else:
            if prob(deltaE, temp) == 1:
                grid_list[I][J] = -1*grid_list[I][J]    #   If Ediff > 0 but exp(-Ediff/temp) < random probability, flip I,Jth dipole
    
    return grid_list



def get_energy(grid_list, L):   #   Takes input 2d list (grid) and returns the total interaction energy of the entire grid
    interaction_energy = 0      #   (essentially does what spin_flip_Ediff does but for the whole grid)
    
    for i in range(L):
        for j in range(L):
            if i == 0:
                top = L - 1 #   If dipole is in the top row, its above neighbour will be in the final row
            else:
                top = i - 1 #   Else its above neighbour is just above
            
            if i == L - 1:
                bottom = 0 #   If dipole is in the bottom row, its below neighbour will be in the first row
            else:
                bottom = i + 1 #   Else its below neighbour is just below

            if j == 0:
                left = L - 1 #   If dipole is in the leftmost column, its left neighbour will be in the final column
            else:
                left = j - 1 #   Else its left neighbour is just to the left

            if j == L - 1:
                right = 0 #   If dipole is in the rightmost column, its right neighbour will be in the first column
            else:
                right = j + 1 #   Else its right neighbour is just to the right

            first = grid_list[int(i)][int(left)]
            second = grid_list[int(top)][int(j)]
            third = grid_list[int(i)][int(right)]  
            fourth = grid_list[int(bottom)][int(j)]   #   Spins of the neighbours of the i-jth dipole (CW from the left)

            if int(grid_list[i][j]) == 1:
                interaction_energy = interaction_energy - (first + second + third + fourth)
            elif int(grid_list[i][j]) == -1:
                interaction_energy = interaction_energy + (first + second + third + fourth)
    
    interaction_energy = 0.5*interaction_energy     #   Factor of 0.5 is needed to account for double-counts of interaction energies!

    return interaction_energy



def get_magnetisation(grid_list, L):    #   Take input 2d list (grid) and outputs the total magnetisation of the grid
    mag = 0

    for i in range(L):
        for j in range(L):
            mag = mag + int(grid_list[i][j])

    return np.abs(mag)



def get_heat_capacity(array, delta_T):  #   Take input 2d list (grid) and outputs the specific heat capacity of the grid
    heat_capacity = [0]

    for i in range(len(array) - 1):
        dif = array[i + 1] - array[i]

        heat_capacity.append(dif / delta_T)

    return np.array(heat_capacity)


    #####################################
    #   #   #   #   #   #   #   #   #   #
    #   #       MAIN PROGRAM        #   #
    #   #   #   #   #   #   #   #   #   #
    #####################################


L = 16   #   The width of the square ising grid

T = 0   #   Temperature, in units of epsilon/k (in order to normalize results) (T !== 0)
dT = 0.15   #   delta T - this will be the interval by which we increase the temperature every time. Choose only values that divide one (e.g. 0.1, 0.2, 0.25 etc.)

amount = 3/dT   # The amount of dTs per unit interval T 
size = 6    # Effectively sets the domain for your T values (dT must divide 1!)

values = int(size*amount / 3) + 1   #   Number of values in our plotted arrays

average_energy_array = np.zeros(values)   #   Best estimate for each T + dT value of energy
average_magnetisation_array = np.zeros(values)    #   Best estimate for each T + dT value of magnetisation
heat_capacity = np.zeros(values)

error_energy_array = np.zeros(values) #   Standard deviation values for each average_energy_array value
error_magnetisation_array = np.zeros(values)  #   Standard deviation values for each average_magnetisation_array value
error_heatcap_array = np.zeros(values)  #   Standard deviation values for each average_magnetisation_array value

dT_array = np.zeros(values)   #   Effectively the x-axis of the plots

counter = 0     #   Arbitrary counter
count = 0
N = 800     #   The amount of representative samples that will be used to calculate the average Energy and Magnetisation values per temperature value

print("Expected operations = " + str((L**2) * 100 * N * values) +
    "      t = " + str(datetime.now())[-15:-4])    #   Indicates how computationally expensive your metropolis algorithm will be
print()



while T <= size + dT:    #   This is where your main program will run. While T is incrementally increased calculate <E>, <M>, and <C>
    energy_array = np.zeros(N)
    magnetisation_array = np.zeros(N)

    for sample in range(N):  #   The amount of representative samples that will be used to calculate the average Energy and Magnetisation values
        grid_list = make_grid(L, T)

        metropolis_algorithm(grid_list, L, T)

        M = get_magnetisation(grid_list, L)
        E = get_energy(grid_list, L)

        magnetisation_array[sample] = M
        energy_array[sample] = E

    print(str(count + 1) +'/' + str(values) + '   t='
          + str(datetime.now())[-15:-4])    #   E.g. output: 21/35   t=21:17:15.75; used to keep track of how many metropolis loops have been made
    
    average_energy_array[count] = np.average(energy_array)
    average_magnetisation_array[count] = np.average(magnetisation_array)

    error_energy_array[count] = np.std(energy_array, ddof=1)
    error_magnetisation_array[count] = np.std(magnetisation_array, ddof=1)

    dT_array[count] = T
    
    count = count + 1

    T = T + dT



heat_capacity = get_heat_capacity(average_energy_array, dT)

fig, (ax_Energy, ax_Magnetisation, ax_HeatCap) = plt.subplots(3, sharex=True)
fig.suptitle("L = " + str(L))
n = L**2

for i in range(values): #   This for-loop scales all averages and their errors to obtain averages and errors per dipole

    average_energy_array[i] = average_energy_array[i] / n
    average_magnetisation_array[i] = average_magnetisation_array[i] / n
    heat_capacity[i] = heat_capacity[i] / n

    if i != values - 1:
        error_heatcap_array[i] = ((1/dT) * np.sqrt(error_energy_array[i+1]**2 + error_energy_array[i]**2)) / n
            #   Adds uncertainty in quadrature and scales by a factor of dT
    else:
        error_heatcap_array[i] = error_energy_array[i] / (dT * n)
        
    error_energy_array[i] = error_energy_array[i] / n
    error_magnetisation_array[i] = error_magnetisation_array[i] / n
    


    #####################################
    #   #   #   #   #   #   #   #   #   #
    #   #       PLOTTING            #   #
    #   #   #   #   #   #   #   #   #   #
    #####################################



#####   Energy subplot
ax_Energy.axvline(2.27, 0, 1, color="red", linestyle=":",
                  label="T = 2.27 $\epsilon$ / $k_B$")
ax_Energy.step(dT_array, average_energy_array, where="mid",
                      color="orange")
ax_Energy.errorbar(dT_array, average_energy_array,
                          yerr=error_energy_array, linestyle="none",
                          color="blue", ecolor="k", marker='s', ms=3,
                   capsize=2)
ax_Energy.set_xlim([0, size])
ax_Energy.set_ylabel("E/N", rotation='vertical')
ax_Energy.annotate("(a)", (0.75, -1), fontsize=10)
ax_Energy.legend()
#####

#####   Magnetisation subplot
ax_Magnetisation.axvline(2.27, 0, 1, color="red", linestyle=":")
ax_Magnetisation.step(dT_array, average_magnetisation_array, where="mid",
                      color="orange")
ax_Magnetisation.errorbar(dT_array, average_magnetisation_array,
                          yerr=error_magnetisation_array, linestyle="none",
                          color="blue", ecolor="k", marker='s', ms=3,
                          capsize=2)
ax_Magnetisation.set_xlim([0, size])
ax_Magnetisation.set_ylabel("M/N", rotation='vertical')
ax_Magnetisation.annotate("(b)", (0.75, 0.5), fontsize=10)
#####

#####   Heat capacity subplot
ax_HeatCap.errorbar(dT_array, heat_capacity,
                          yerr=error_heatcap_array, linestyle="none",
                          color="blue", ecolor="k", marker='s', ms=3,
                   capsize=2)
ax_HeatCap.step(dT_array, heat_capacity, where="mid",
                      color="orange")
ax_HeatCap.axvline(2.27, 0, 1, color="red", linestyle=":")
ax_HeatCap.set_xlim([0, size])
ax_HeatCap.set_xlabel('Temperature ($\epsilon$ / $k_B$)')
ax_HeatCap.set_ylabel("C/N := $\Delta$E/$\Delta$T", rotation='vertical')
ax_HeatCap.annotate("(c)", (0.75, np.max(heat_capacity)), fontsize=10)
#####
plt.show()
