import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import datetime

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#     :: Constants ::

Nb = 101 #Number of cells
L  = 3E24 #Length of the simulation
dx = L/Nb #Length of each cell
dt = 1E11 #Time step in sec
t  = 1E14 #Total time in sec
NbIterations = int(t/dt) #Number of time iterations
c  = 3E10 #Speed of light

T     = 8000
rho   = 2
sigma = 1.63e-18
x     = 1e-4  #Initial fraction of hydrogen


now = datetime.datetime.now() #In order to save the plots with different names (will give the hour, see class Plot)


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#     :: Classes ::

class Calculation():
    #Constructor::____________________________________________________________________________________________________
    def __init__(self, FHalf, PHalf, N, P, F, T, rho, dt, sigma, c, x, WithOrWithout):
        self.FHalf = FHalf
        self.PHalf = PHalf

        self.N = N
        self.P = P
        self.F = F

        ObjChemistry = Chemistry(T)
        Rates        = ObjChemistry.Get_Rates()
        self.alphaAH = Rates[0]
        self.alphaBH = Rates[1]
        self.betaH   = Rates[2]

        self.rho   = rho
        self.dt    = dt
        self.sigma = sigma
        self.c     = c

        self.HydroFraction   = [[]]
        self.TotalIonisation = []
        self.NbIonisedCells  = []

        for i in range(NbIterations): #Creates as many locations to store the data from the "Nb" cells for the "NbIterations" iterations
            if i == 0:
                for I in range(Nb):
                    self.HydroFraction[0].append(x)
            else:
                self.HydroFraction.append(np.zeros(Nb))

        self.Condition = WithOrWithout

    #Getter::_________________________________________________________________________________________________________
    def Get_HydroFraction(self):
        return self.HydroFraction

    #Methods::________________________________________________________________________________________________________
    def GLF(self, i, time):
        if i == 0: #Boundary conditions
            if time == 0:
                self.FHalf[time][i] = 0 #The zeros comes from the inital conditions
                self.PHalf[time][i] = 0 #in the last cell at t = 0
            else: #For the values of PHalf_{n \pm 1/2} and FHalf_{n \pm 1/2}... COMPLETE
                self.FHalf[time][i] = (1/2)*(self.F[time - 1][Nb - 1] + self.F[time][i]) - (self.c/2)*(self.N[time][i] - self.N[time - 1][Nb - 1])
                self.PHalf[time][i] = (1/2)*(self.P[time - 1][Nb - 1] + self.P[time][i]) - (self.c/2)*(self.F[time][i] - self.F[time - 1][Nb - 1])

        else:
            self.FHalf[time][i] = (1/2)*(self.F[time][i - 1] + self.F[time][i]) - (self.c/2)*(self.N[time][i] - self.N[time][i - 1])
            self.PHalf[time][i] = (1/2)*(self.P[time][i - 1] + self.P[time][i]) - (self.c/2)*(self.F[time][i] - self.F[time][i - 1])

    def NFP(self, i, time):
        if time == (NbIterations - 1):
            if i == (Nb - 1):
                print("The integration has been performed, please wait for the generation of the graphs you asked for.")

        else:
            if i == (Nb - 1): #Boundary condition, if we are at the "last" cell during a time iteration
                self.N[time + 1][i] = self.N[time + 1][i] + self.N[time][i] - (dt/dx)*(self.FHalf[time][0] - self.FHalf[time][i]) #Calculation of the value of N_{n+1}
                if self.N[time + 1][i] == 0:
                    self.N[time + 1][i] = 10E-21
                self.F[time + 1][i] = self.F[time][i] - (dt/dx)*(self.PHalf[time][0] - self.PHalf[time][i]) #Calculation of the value of F_{n+1}

                if self.Condition == 1:
                    ChemistrySol = self.ChemistrySolution(self.N[time + 1][i], self.F[time + 1][i], time, i)
                    self.N[time + 1][i] = ChemistrySol[0]
                    self.F[time + 1][i] = ChemistrySol[1]

                f                   = (self.F[time + 1][i])/(c*self.N[time + 1][i]) #Calculates the value of the reduced flux
                Chi                 = (3 + 4*f*f)/(5 + 2*np.sqrt(4 - 3*f*f)) #Calculates the value of chi
                self.P[time + 1][i] = Chi*self.N[time + 1][i]*c*c #Deduction of the value of P_{n+1}

            else:
                self.N[time + 1][i] = self.N[time + 1][i] + self.N[time][i] - (dt/dx)*(self.FHalf[time][i+1] - self.FHalf[time][i]) #Calculation of the value of N_{n+1}
                if self.N[time + 1][i] == 0:
                    self.N[time + 1][i] = 10E-21
                self.F[time + 1][i] = self.F[time][i] - (dt/dx)*(self.PHalf[time][i+1] - self.PHalf[time][i]) #Calculation of the value of F_{n+1}

                if self.Condition == 1:
                    ChemistrySol = self.ChemistrySolution(self.N[time + 1][i], self.F[time + 1][i], time, i)
                    self.N[time + 1][i] = ChemistrySol[0]
                    self.F[time + 1][i] = ChemistrySol[1]

                f                   = (self.F[time + 1][i])/(c*self.N[time + 1][i]) #Calculates the value of the reduced flux
                Chi                 = (3 + 4*f*f)/(5 + 2*np.sqrt(4 - 3*f*f)) #Calculates the value of chi
                self.P[time + 1][i] = Chi*self.N[time + 1][i]*c*c #Deduction of the value of P_{n+1}


    def ThirdOrderPoly(self, N, time, cell):
        x = self.HydroFraction[time][cell] #Take the current fraction of ionised hydrogen atoms

        m = (self.alphaBH + self.betaH)*(self.rho**2)*self.dt
        n = self.rho - ((self.alphaAH + self.betaH)*self.rho)/(self.sigma*self.c) - (self.rho**2)*self.dt*(self.alphaBH + 2*self.betaH)
        p = - self.rho*(1 + x) - N - 1/(self.sigma*self.c*self.dt) + (self.betaH*self.rho)/(self.sigma*self.c) + self.betaH*(self.rho**2)*self.dt
        q = N + self.rho*x + x/(self.sigma*self.c*self.dt)

        root = np.roots((m,n,p,q))
        for i in root:
            if i > 0 and i < 1:
                self.HydroFraction[time + 1][cell] = i #New fraction value added to the list
                

    def ChemistrySolution(self, N_advective, F_advective, Time, Cell):
        self.ThirdOrderPoly(N_advective, Time, Cell)
        PreviousX = self.HydroFraction[Time][Cell]#self.ThirdOrderPoly(N_advective)
        CurrentX  = self.HydroFraction[Time + 1][Cell]

        N = N_advective + self.betaH*(self.rho**2)*(1 - CurrentX)*CurrentX*self.dt - self.alphaBH*(self.rho**2)*(CurrentX**2)*self.dt - self.rho*(CurrentX - PreviousX)
        F = F_advective / (1 + self.c*self.sigma*self.rho*self.dt*(1 - CurrentX))

        return N, F
        
    def SumIonisation(self):
        for i in range(len(self.HydroFraction)):
            self.TotalIonisation.append(sum(self.HydroFraction[i]))
            count = 0
            for j in range(Nb):
                if self.HydroFraction[i][j] >= 0.9:
                    count += 1
                    
            self.NbIonisedCells.append(count/Nb)
            
        return self.TotalIonisation, self.NbIonisedCells

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

class Chemistry():
    def __init__(self, T):
        self.T     = T

    def Get_Rates(self):
        return self.alphaAH(), self.alphaBH(), self.betaH()

    def alphaAH(self):
        Lambda = (2*157807)/(self.T)
        res    = 1.269E-13
        res   *= pow(Lambda, 1.503)
        res   /= pow(1 + (Lambda/0.522)**0.47, 1.923)
        return res

    def alphaBH(self):
        Lambda = (2*157807)/(self.T)
        res    = 2.753E-14
        res   *= pow(Lambda, 1.5)
        res   /= pow(1 + (Lambda/2.74)**0.407, 2.242)
        return res

    def betaH(self):
        Lambda = (2*157807)/(self.T)
        res    = 21.11*pow(self.T, (-3/2))*np.exp(-Lambda/2)*pow(Lambda, -1.089)
        res   /= pow(1 + pow(Lambda/0.354, 0.874), 1.01)
        return res


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

class StellarSources():
    #Constructor::____________________________________________________________________________________________________
    def __init__(self, Chemistry, T):
        #Let's take an example of how self.N, self.P and self.F will be defined (see Initialise method below)::
        #self.N[0][1] will contain the information about the number density of photons in the cell 1 at time dt = 0 (0th time iteration)
        self.N = [] #Values of the photons number density
        self.P = [] #Values of the pressure
        self.F = [] #Values of the flux

        self.PHalf = [] #Values requiered for the calculation of P^{n+1}
        self.FHalf = [] #Values requiered for the calculation of F^{n+1}

        for i in range(NbIterations): #Creates as many locations to store the data from the "Nb" cells for the "NbIterations" iterations
            self.N.append(np.zeros(Nb)) #Values of the photons number density
            self.P.append(np.zeros(Nb)) #Values of the pressure
            self.F.append(np.zeros(Nb)) #Values of the flux

            self.PHalf.append(np.zeros(Nb)) #Values requiered for the calculation of P^{n+1}
            self.FHalf.append(np.zeros(Nb)) #Values requiered for the calculation of F^{n+1}

        self.Chem     = Chemistry
        self.Fraction = []
        self.SumFract = []
        
        self.T = T

    #Getter::_________________________________________________________________________________________________________
    def Get_Data(self):
        return self.N, self.Fraction, self.SumFract

    #Methods::________________________________________________________________________________________________________
    def PulseSource(self): #Initial condition on the source, 1 photon emitted at 0th iteration from the cell 0
        #Initial conditions::
        self.N[0][int(Nb/2)] = 100

    def ContinuousSource(self): #Initial condition on the source, 1 photon emitted at 0th iteration from the cell 0
        #Initial conditions::
        for Time in range(NbIterations):
            self.N[Time][int(Nb/2)] = 10

    def DoubleSource(self): #Initial condition on the source, 1 photon emitted at 0th iteration from the cell 0
        #Initial conditions::
        for Time in range(NbIterations):
                self.N[Time][int(Nb/3)] = 1
                self.N[Time][int(2*(Nb/3))] = 1

    def SourcePacket(self): #Initial condition on the source, 1 photon emitted at 0th iteration from the cell 0
        #Initial conditions::
        for Time in range(NbIterations):
            self.N[Time][int(Nb/2)] = 1
            for i in range(1,6):
                self.N[Time][int(Nb/2) + i] = 1
            for i in range(1,6):
                self.N[Time][int(Nb/2) - i] = 1

    def SelectSource(self, WhichSource): #Initial condition on the source, 1 photon emitted at 0th iteration from the cell 0
        if WhichSource == 1:
            self.PulseSource()
        if WhichSource == 2:
            self.ContinuousSource()
        if WhichSource == 3:
            self.DoubleSource()
        if WhichSource == 4:
            self.SourcePacket()

        #Calculations using the methods from the "Integation" class::
        A = Calculation(self.FHalf, self.PHalf, self.N, self.P, self.F, self.T, rho, dt, sigma, c, x, self.Chem) #Creates an object A to integrate

        for Time in range(NbIterations):
            for Cell in range(Nb):
                A.GLF(Cell, Time)
            for Cell in range(Nb):
                A.NFP(Cell, Time)

        self.Fraction = A.Get_HydroFraction()
        self.SumFract = A.SumIonisation()[0]
       


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

class Plot():
    #Constructor::____________________________________________________________________________________________________
    def __init__(self, YAxis1, YAxis2):
        self.Y1 = YAxis1
        if YAxis2 == 0:
            self.Y2 = 0
        else:
            self.Y2 = YAxis2

    #Methods::________________________________________________________________________________________________________
    #Private method::
    def __LightPath(self, d): #This private method calculates the distance travelled by light based on the iteration and c.
        PathTime, CellCount = 0, 0
        DistNthCell = dx
        for i in range(NbIterations):
            Time = dt * i
            if (c*Time) >= DistNthCell:
                CellCount += 1
                d[i] = CellCount + int(Nb/2)
                DistNthCell = dx*(CellCount + 1)
            else:
                d[i] = CellCount + int(Nb/2)
        return d

    #Public methods
    def DensityInACell(self, CellNumber): #We can choose in which cell we want to see the evolution of N
        XAxis = np.arange(NbIterations)
        YAxis = []
        for i in range(NbIterations):
            YAxis.append(self.Y1[i][int(Nb/2)])

        plt.plot(XAxis, YAxis, '-ok', c="red")
        plt.title('Photons number density as a function of the iteration in the cell number' + str(CellNumber), fontname = 'Serif', size = 13)
        plt.xlabel('Time', fontname = 'Serif', size = 11)
        plt.ylabel('Value of N in the 0th cell', fontname = 'Serif', size = 11)
        plt.show()

    def AllGraphsInARow(self): #Prints all graphs for the density of photons in a row, scrolling skills will be requiered
        XAxis = np.arange(Nb)

        Light = np.zeros(NbIterations)
        Light = self.__LightPath(Light)

        for i in range(NbIterations):
            plt.plot(XAxis, self.Y1[i], '-ok', c="red")
            plt.axvline(Light[i])
            #plt.title('Photons number density as a function of the iteration in the cell number' + str(Nb), fontname = 'Serif', size = 13)
            #plt.xlabel('Time', fontname = 'Serif', size = 11)
            #plt.ylabel('Value of N in the 0th cell', fontname = 'Serif', size = 11)
            plt.show()
            
    def Fraction(self, X, Y, T):
        for i in range(5):
           plt.plot(X, Y[i], label=str(T[i]) + "K")
        plt.legend()   
        plt.show()

    def Animation(self): #Animates the evolution of the density as a function of time
    	#for i in range(NbIterations - 5):
            #print(abs(self.F[i][int(Nb/2) - 3]) - abs(self.F[i][int(Nb/2) + 3]))

        XAxis = np.arange(Nb)
        fig,ax = plt.subplots()

        #Light = np.zeros(NbIterations)
        #Light = self.__LightPath(Light)

        def animate(i):
            #YMax = np.amax(self.N1[i] + self.N2[i])
            ax.clear()
            #ax.text((Nb - 16), (YMax), 'Time:: ' + str(round(i*dt, 2)) + 's', fontname = 'Serif', size = 11)
            ax.set_title('Number density in each cell as a function of time ', fontname = 'Serif', size = 13)
            ax.set_xlabel('Cell number', fontname = 'Serif', size = 11)
            ax.set_ylabel('Density of photons', fontname = 'Serif', size = 11)
            #ax.set_xlim(0, Nb)
            #ax.set_ylim(0, YMax)
            line, = ax.plot(XAxis, self.Y1[i], c="red", label="Without Chemistry")
            if self.Y2 == 0:
                return line,
                #line1, = ax.plot(Light[i], YMax/2, marker  = ">", c="gold")
            else:
                line1, = ax.plot(XAxis, self.Y2[i], c="black", label="With Chemistry")
                ax.legend(handles=[line, line1])
                return line, line1,

        ani = FuncAnimation(fig, animate, interval=1, blit=True, repeat=True, frames=1500)
        ani.save("Evolution_NbDensity" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + ".gif", dpi=300, writer=PillowWriter(fps=25))


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#     :: Main ::


#At cell number 0 we create an object Integation.
#The initial conditions in that case are that at t=0, the pressure, the density and the flux in any other cell than the 0th must be null
#So we need:: N[0] = 0 ; P[0] = 0 ; F[0] = 0

print(3*c*dt/dx)
if 3*c*dt/dx < 1:
    print("The courant condition is respected, the program can proceed.")
    #StelObj1 = StellarSources(1) #Without chemistry
    #StelObj1.SelectSource(2)
    #Y1       = StelObj1.Get_Data()[1] #0 for N and 1 for fraction
    #StelObj2 = StellarSources(1) #With chemistry
    #StelObj2.SelectSource(1)
    #Y2       = StelObj2.Get_Data()[0]

    #PlotObj = Plot(Y1, 0)
    #PlotObj.Animation()
    #PlotObj.AllGraphsInARow()
    
    T = [1000, 4000, 8000, 15000, 21000]
    X = np.linspace(0, t, NbIterations)
    Y = []
    for i in range(len(T)):
        StelObj = StellarSources(1, T[i])
        StelObj.SelectSource(2)
        Y.append(StelObj.Get_Data()[2])

    PlotObj = Plot(Y, 0)
    PlotObj.Fraction(X, Y, T)
    
else:
    print("The courant condition is not respected, please try other values.\nPlease try to enter other initial conditions.")



#Limbo
