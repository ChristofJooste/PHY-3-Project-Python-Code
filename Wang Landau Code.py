from datetime import datetime
import numpy as np
import numpy.random as random
import math
import matplotlib.pyplot as plt
from numpy import inf

 #####################################
 #   #   #   #   #   #   #   #   #   #
 #   #   FUNCTIONS & PROCEDURES  #   #
 #   #   #   #   #   #   #   #   #   #
 #####################################


def rand_zero_one():  # Function to obtain a random value between 0 and 1
    R = random.uniform(0, 1, 1)
    # Returns (1) random number in the range [1E-50,1) to prevent overflow in logarithm values
    return np.round(R, 5)


def make_grid(L):  # Creates an L x L Ising grid with random spin-up/ spin-down elements
    # Creates a 2-d list of size L x L, populated with only ones (makes the entire matrix spin-up)
    grid = [[1 for i in range(L)] for j in range(L)]

    for i in range(L):  # Row number
        for j in range(L):  # Column number
            if (i + j) % 2 == 0:  # Sets random matrix elements to spin-down elements (i.e. you end up with a randomized L x L ising matrix)
                grid[i][j] = -1
            else:
                grid[i][j] = 1

    return grid


# Function to obtain random row or column in L x L Ising grid.
def rand_row_col(L):
    R = random.randint(0, L, 1)  # Returns (1) integer in  the range [0, L)
    return int(R)


# L must be EVEN! Calculates and returns the allowed energy states of the entire Ising grid
def energy_spectrum(L):

    d = 4

    N = L**2
    E = -2*N
    E_list = []

    while E <= 2*N:
        E_list.append(E)
        # Creates E_allowed = {-2N, -2N+4, -2N+8, ... , +2N-8, +2N-4, +2N}
        E = E + d

    E_list.pop(1)  # Removes the -2N+4 entry from E_allowed
    E_list.pop(-2)  # Removes the +2N-4 entry from E_allowed

    return E_list


# Defines a visits histogram on the spectrum of allowed energy states
def make_histogram(energy_spectrum):
    histogram = []

    for energy in energy_spectrum:
        # Populates the histogram with all 0 entries (no visits have been made)
        histogram.append(0)

    return histogram


# Defines a set of degeneracies on the spectrum of allowed energy states
def make_degeneracies(energy_spectrum):
    degeneracies = []

    for energy in energy_spectrum:
        degeneracies.append(np.log(1))

    return degeneracies


# Takes input 2d list (grid) and returns the total interaction energy of the entire grid
def get_energy(grid_list, L):
    interaction_energy = 0

    for i in range(L):
        for j in range(L):
            if i == 0:
                top = L - 1  # If dipole is in the top row, its above neighbour will be in the final row
            else:
                top = i - 1  # Else its above neighbour is just above

            if i == L - 1:
                bottom = 0  # If dipole is in the bottom row, its below neighbour will be in the first row
            else:
                bottom = i + 1  # Else its below neighbour is just below

            if j == 0:
                left = L - 1  # If dipole is in the leftmost column, its left neighbour will be in the final column
            else:
                left = j - 1  # Else its left neighbour is just to the left

            if j == L - 1:
                right = 0  # If dipole is in the rightmost column, its right neighbour will be in the first column
            else:
                right = j + 1  # Else its right neighbour is just to the right

            first = grid_list[int(i)][int(left)]
            second = grid_list[int(top)][int(j)]
            third = grid_list[int(i)][int(right)]
            # Spins of the neighbours of the i-jth dipole, clockwise from the left neighbour
            fourth = grid_list[int(bottom)][int(j)]

            if int(grid_list[i][j]) == 1:
                interaction_energy = interaction_energy - \
                    (first + second + third + fourth)
            elif int(grid_list[i][j]) == -1:
                interaction_energy = interaction_energy + \
                    (first + second + third + fourth)

    # Factor of 0.5 needed to account for double counts of interaction energies
    interaction_energy = 0.5*interaction_energy

    return interaction_energy



def sig_fig(number):    #   E.g. Input = 1.73765E+32; Output = 1.7377

    scientific_notation = "{:e}".format(number)

    temp = scientific_notation.split("e")

    return float(temp[0])



def order_of_magnitude(number): #   E.g. Input = 1.73765E+32; Output = +32

    scientific_notation = "{:e}".format(number)

    temp = scientific_notation.split("e")
    
    return int(temp[1])



def approximate_exponential(number, order_correction):  #   Attempts to approximate e^x for "large" x according to an n-th order summation of the definition of exp(x)

    if number > -150 and number < 150: 

        return math.exp(number)

    elif number <= -150:

        return 0

    elif number >= 150:
        
        sum_terms = 0

        for i in range(order_correction + 1):

            sum_terms = sum_terms + (number**i) / math.factorial(i)

        return sum_terms



def decompose_exp_log(log_number):  # calculates e^ln(x) = x for "large" x (of order >75)
    factor = math.floor(log_number / 100)

    if factor == 0:
        return math.exp(log_number)
    else:

        remainder = log_number - factor*100

        # E.g. if ln(g) = 702.56; g = e^ln(g) = 7e^100 * e^2.56
        return (math.exp(100))**factor * math.exp(remainder)



def decompose_logarithm(number):  # calculates ln(x) for "large" x (of order >75)

    o_o_m = order_of_magnitude(number)

    factor = sig_fig(number)

    return np.log(factor) + o_o_m*math.log(10)



def prob(energy_spectrum, E_one, E_two, degeneracies): #   Returns the probability factor to accept or reject
                                                       #   the proposed spin-flip from energy states E_one to E_two
    pos_one = energy_spectrum.index(E_one)
    pos_two = energy_spectrum.index(E_two)

    deg_one = degeneracies[pos_one]
    deg_two = degeneracies[pos_two]

    ln_ratio = deg_one - deg_two

    # Returns the minimum between our ratio and 1 using a log scale to prevent overflow
    return min(ln_ratio, 0)



def attempt_flip(grid_list, L, energy_spectrum, degeneracies, histogram, f):    #   Attempts a spin flip that will transform the grid
                                                                                #   between two different allowed energy states
    I = rand_row_col(L)
    J = rand_row_col(L)
    try:
        ln_R = math.log(float(rand_zero_one()))
    except ValueError:
        ln_R = -inf

    # Trial grid where the dipole in question is flipped to calculate the
    trial_grid = [[0 for i in range(L)] for j in range(L)]
                                                         #   energy state (trial flip)
    for i in range(L):
        for j in range(L):
            if i == I and j == J:
                trial_grid[i][j] = -grid_list[i][j]
            else:
                trial_grid[i][j] = grid_list[i][j]

    energy_before = get_energy(grid_list, L)
    energy_after = get_energy(trial_grid, L)

    ln_P = float(prob(energy_spectrum, energy_before,
                 energy_after, degeneracies))
    # If the spin flip is rejected, use this index to update g(E1) and H(E1)
    index = energy_spectrum.index(energy_before)

    # If the probability factor = 1 or is less that some randomly generated float between 0 and 1, accept the proposed spin flip
    if ln_P == 0 or ln_R == -inf or ln_R <= ln_P:
        for i in range(L):
            for j in range(L):
                grid_list[i][j] = trial_grid[i][j]

        # If the spin flip is accepted, use this index to update g(E2) and H(E2)
        index = energy_spectrum.index(energy_after)

    # Update visits histogram at the relevant energy state
    histogram[index] += 1
    # Update the set of degeneracies at the relevant energy state
    degeneracies[index] = np.round(degeneracies[index] + np.log(f), 5)
                                                                     #   (log scale is used to prevent overflow)


def flat(histogram, x):  # Sees if histogram qualifies as "flat" based off of the criteria that all H(E) values are
                     #   within (1-x) of the average <H(E)> value
    average = sum(histogram) / len(histogram)
    minimum = x*average
    maximum = (2 - x)*average
    flag = 1  # Assume the histogram is flat

    for i in histogram:  # Attempt to disprove flatness
        if i < minimum or i > maximum:
            flag = 0

    return flag


# Calculates the partition function for some temperature
def partition(temp, energy_spectrum, log_degeneracies):

    sum_E_states = 0

    for i, E in enumerate(energy_spectrum):
        decomposed_one = decompose_exp_log(log_degeneracies[i])
        decomposed_two = approximate_exponential(-E/temp, 20)

        sum_E_states = sum_E_states + decomposed_one*decomposed_two

    return math.exp(-temp) * sum_E_states


# Calculates <E> for some given temperature
def average_energy(temp, energy_spectrum, log_degeneracies):

    numerator = 0
    denominator = 0

    for i, E in enumerate(energy_spectrum):
        decomposed_one = decompose_exp_log(log_degeneracies[i])
        decomposed_two = approximate_exponential(-E/temp, 20)

        numerator = numerator + E*decomposed_one*decomposed_two
        # Same as sum_E_states for partition(...)
        denominator = denominator + decomposed_one*decomposed_two

    return numerator/denominator


def get_heat_capacity_array(dT, average_energy_array):

    heat_capacity = [0, 0]

    for i in range(len(average_energy_array) - 2):  # Prevents division by 0
        dif = average_energy_array[i + 2] - average_energy_array[i + 1]

        heat_capacity.append(dif / dT)

    return np.array(heat_capacity)


def get_free_energy(temp, energy_spectrum, log_degeneracies):

    Z = partition(temp, energy_spectrum, log_degeneracies)
    #print(Z)
    decomposed_log = decompose_logarithm(Z)

    F = -1*temp*decomposed_log

    return F


def get_entropy_array(dT, free_energy_array):

    entropy = [math.log(2)]

    for i in range(len(free_energy_array) - 1):  # Prevents division by 0
        dif = free_energy_array[i + 1] - free_energy_array[i]

        entropy.append(-1*dif / dT)

    return np.array(entropy)


    #####################################
    #   #   #   #   #   #   #   #   #   #
    #   #       MAIN PROGRAM        #   #
    #   #   #   #   #   #   #   #   #   #
    #####################################


L = 4  # Size of Ising grid
# Flatness criterion. "Flatness" is defined as within 1-x of histogram average for all values of H(E)
x = 0.9
cutoff = 1 + 1E-8  # Once f < this value, stop the program. Cutoff = 1.00000001

energy_spectrum = energy_spectrum(L)

# Note: this is also exactly the same as len(histogram) and len(degeneracies)
energies = len(energy_spectrum)

iteration = 0  # Arbitrary counting variable to keep track of code progress
counter = -1  # Arbitrary counting variable to keep track of code progress

# Number of repeats with which to perform Type A uncertainty analysis. For comparison and justification purposes.
repeat = 1
temporary_degeneracy_array = [[0 for a in range(energies)] for b in range(repeat)]

for j in range(repeat):

    counter += 1
    print("Begin W-L run    t=" + str(datetime.now())[-15:-4])

    histogram = make_histogram(energy_spectrum)
    degeneracies = make_degeneracies(energy_spectrum)

    grid_list = make_grid(L)

    # Modification factor for degeneracy value - gets updated to f = e^1 = 2.718.... below (when the WL algorithm starts)
    f = np.exp(2)

    # Equivalent to a repeat .. until loop: Runs the program until certain criteria are met (f < 1E-8)
    while True:

        f = np.sqrt(f)  # Adjusts f by square rooting it
        counts = 0  # Arbitrary counting variable

        if f < cutoff:
            print("Cutoff Reached.")
            iteration = 0
            break

        # Equivalent to a repeat .. until loop: Runs the program until certain criteria are met (Histogram = "Flat")
        while True:
            attempt_flip(grid_list, L, energy_spectrum,
                         degeneracies, histogram, f)
            counts += 1
            # Check for histogram flatness after every 10 000 proposed flips
            if counts % 10000 == 0 and flat(histogram, x) == 1:
                # If histogram is flat, reset the visits histogram to all 0 and update f = sqrt(f)
                for i in range(energies):
                    histogram[i] = 0

                break

        print("Iteration " + str(iteration + 1) + "/27 completed.   t=" + \
              str(datetime.now())[-15:-4])  # Keeps track of the iteration you are on
        iteration += 1

    for c in range(energies):
        # 2-d list to be used in uncertainty propagation
        temporary_degeneracy_array[counter][c] = degeneracies[c]

    #############################################
    #   #   #   #   #   #   #   #   #   #   #   #
    #   #       DEALING WITH LOGARITHMS     #   #
    #   #   #   #   #   #   #   #   #   #   #   #
    #############################################


average_degeneracy_array = np.zeros(energies)
error_degeneracy_array = np.zeros(energies)

ln_average_degeneracy_array = np.zeros(energies)

correction_factors_array = []


for r in range(repeat):  # Fixes row number
    # Ensures all first entries in temporary_degeneracy_array are normalized to ln(2)
    K = temporary_degeneracy_array[r][0] - np.log(2)

    correction_factors_array.append(K)

    # temporary_corrected_array[r]

    for g in range(energies):  # Corrects each column number
        # Corrects value and goes from log scale to linear scale for Type A evaluation of uncertainty
        temporary_degeneracy_array[r][g] = np.exp(temporary_degeneracy_array[r][g] - correction_factors_array[r])

#comparison_array = np.zeros(energies)

for g in range(energies):  # Fixes column number

    temporary_sum_array = np.zeros(repeat)

    for r in range(repeat):  # Adds up row numbers

        temporary_sum_array[r] = temporary_degeneracy_array[r][g]

    average_degeneracy_array[g] = np.average(temporary_sum_array)  # Note: these are on linear scales

    # Note: these are on logarithmic scales
    ln_average_degeneracy_array[g] = np.log(average_degeneracy_array[g])


    #####################################
    #   #   #   #   #   #   #   #   #   #
    #   #           PLOTTING        #   #
    #   #   #   #   #   #   #   #   #   #
    #####################################


T = np.linspace(0, 6, 60)
dT = T[1] - T[0]

En = []
F = []
Z_array = []
uZ_array = []


for j, temp in enumerate(T):
    En.append(average_energy(temp, energy_spectrum, ln_average_degeneracy_array))

    F.append(get_free_energy(temp, energy_spectrum, ln_average_degeneracy_array))


En = np.array(En)
F = np.array(F)
C = get_heat_capacity_array(dT, En)
S = get_entropy_array(dT, F)

flag = False

while flag == False:
    for i in range(energies):
        if En[i] <= En[i + 1] and En[i+2] <= En[i+3]:
            flag = True
            break

fig, ((ax_E, ax_F), (ax_C, ax_S)) = plt.subplots(
    2, 2, sharex='col', figsize=[6, 5])
fig.tight_layout(pad=2.0)

plt.suptitle("L = " + str(L) + ", cutoff: f < " + str(cutoff) +
             ", flatness criterion: x = " + str(int(100*x)) + "%")

ax_E.axvline(2.27, 0, 1, linestyle="--", color='r',
             label="T = 2.27 $\epsilon$ / $k_B$")
ax_E.plot(T, En/L**2, color='blue')
ax_E.set_ylabel('E/N')
ax_E.set_xlim([T[i], 6])
ax_E.set_ylim([-2, 0])
ax_E.annotate("(a)", (1, -1))
ax_E.legend()

ax_F.axvline(2.27, 0, 1, linestyle="--", color='r')
ax_F.plot(T, F/L**2, color='orange')
ax_F.set_ylabel('F/N')
ax_F.set_xlim([T[i], 6])
ax_F.set_ylim([F[58]/L**2 - 0.5, -1.5])
ax_F.annotate("(b)", (1, -3.5))

ax_C.axvline(2.27, 0, 1, linestyle="--", color='r')
ax_C.plot(T, C/L**2, color='green')
ax_C.set_ylabel('C/N := ∂E/∂T')
ax_C.set_ylim([0, C[23]/L**2 + 0.5])
ax_C.annotate("(c)", (4.5, C[23]/L**2 - 0.5))

ax_S.axvline(2.27, 0, 1, linestyle="--", color='r')
ax_S.plot(T, S/L**2, color='black')
ax_S.set_ylabel('S/N := ∂F/∂T')
ax_S.set_ylim([0, S[58]/L**2 + 0.2])
ax_S.annotate("(d)", (4, S[58]/L**2 - 0.2))

plt.show()