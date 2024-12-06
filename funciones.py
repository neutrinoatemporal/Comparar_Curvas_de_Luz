import numpy as np

def newton_raphson(g, dg, x0, tol, max_iter=100): # Método de Newton-Raphson
    x = x0
    for i in range(max_iter):
        gx = g(x)
        dgx = dg(x)
        if abs(gx) < tol:  # Si el valor de g(x) es suficientemente pequeño, se ha encontrado la raíz
            return x  # Devuelve la raíz y el número de iteraciones
        if dgx == 0:  # Si la derivada es cero, no se puede aplicar el método
            raise ValueError("Derivada cero. No se puede continuar con Newton-Raphson.")
        x = x - gx / dgx  # Fórmula del método de Newton-Raphson
    raise ValueError("El método no convergió en el número máximo de iteraciones.")

def rp(q,p): # Radio Polar

    def f(r):
        return 1/r + q*(1/(np.sqrt(r**2+1)))- p

    def df(r):
        return -1/r**2- q*r/(r**2+1)**(3/2)

    def g(x):
        return -1/x**2 - q*(((x-1)/np.sqrt(x**2-2*x+1)**3)+1)+ (1+q)*x

    def dg(x):
        return 2/x**3+ 2*q/np.sqrt(x**2-2*x+1)**3 + (1+q)
    tol = 1e-9
    r0 = newton_raphson(g, dg, 0.5, tol,100)/3
    raiz = newton_raphson(f, df, r0, tol,100)

    return raiz

#Área de segmentos
def areaseg(R,phi):
    return (R**2)*(phi-np.sin(phi))/2

#Area1
def area1(ro,r1,r2,z1,z2,phi1,phi2):

    if (ro>=r1+r2): #No hay eclipse
        if(z1>z2):
            A1= np.pi*r1**2
        else:
            A1= np.pi*r1**2
    elif (ro<=r1-r2): #eclipse total
        if(z1>z2):
            A1= np.pi*r1**2
        else:
            A1= np.pi*r1**2 - np.pi*r2**2
    else: #Eclipse parcial
        if(z1>z2):
            A1= np.pi*r1**2
        else:
            A1= np.pi*r1**2- areaseg(r1,phi1)-areaseg(r2,phi2)
    return A1

#Área2
def area2(ro,r1,r2,z1,z2,phi1,phi2):

    if (ro>=r1+r2): #No hay eclipse
        if(z1>z2):
            A2= np.pi*r2**2
        else:
            A2= np.pi*r2**2
    elif (ro<=r1-r2): #eclipse total
        if(z1>z2):
            A2= 0
        else:
            A2= np.pi*r2**2
    else: #Eclipse parcial
        if(z1>z2):
            A2= np.pi*r2**2- areaseg(r1,phi1)-areaseg(r2,phi2)
        else:
            A2= np.pi*r2**2
    return A2

#Radio aparente
def ra(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 )

#Ángulo phi1 para área de segmentos
def phiuno(r1, r2, ra):
    # Asegurarse de que no haya división por cero en ningún valor de ra
    if r1 == 0 or ra == 0:  # Si alguno de los valores en 'ra' es 0
        return 0  # O algún valor apropiado según el contexto
    # Calcular el argumento de arccos
    argument = (r1**2 + ra**2 - r2**2) / (2 * r1 * ra)
    # Limitar el argumento entre -1 y 1 para evitar errores en arccos
    argument = np.clip(argument, -1, 1)
    # Devolver el valor de phiuno
    return 2 * np.arccos(argument)

#Ángulo phi2 para área de segmentos
def phidos(r1, r2, ra):
    # Asegurarse de que no haya división por cero en ningún valor de ra
    if r1 == 0 or ra == 0:  # Si alguno de los valores en 'ra' es 0
        return 0  # O algún valor apropiado según el contexto
    # Calcular el argumento de arccos
    argument = (r2**2 + ra**2 - r1**2) / (2 * r2 * ra)
    # Limitar el argumento entre -1 y 1 para evitar errores en arccos
    argument = np.clip(argument, -1, 1)
    # Devolver el valor de phiuno
    return 2 * np.arccos(argument)