__precompile__()

module herramientas

using LaTeXStrings

export metodo_newton, metodo_rectangulo, metodo_trapecio, metodo_simpson, interpolador_lagrange, euler, euler_implicito, RK4

function derivada_numerica(f,x0)
    
    return (9*f(x0+5*0.01)-125*f(x0+3*0.01)+2250*f(x0+0.01)-2250*f(x0-0.01)+125*f(x0-3*0.01)-9*f(x0-5*0.01))/(3840*0.01)
    
end

function punto_fijo(f,x0)
    
    x = x0
    for i in 1:20
        x = f(x)
    end
    
    return x
    
end

"""Estima el valor de las raíces de una función ``\\:f``, i.e. los valores de ``x`` para los cuales ``\\:f(x)=0``, utilizando el método de Newton, `metodo_newton```(\\:f,\\mathbf{x}_{0})`` necesita de dos argumentos: la función ``\\:f`` y una condición inicial ``\\mathbf{x}_{0}`` la cual entre más cercana sea a una raíz de ``\\:f``, la convergencia del método será más rápida. Dicha condición inicial ``\\mathbf{x}_{0}`` convergerá a la raíz de ``\\:f`` que se encuentre más cerca. Además ``\\mathbf{x}_{0}`` puede ser un vector y la función regresará un vector con raíces de ``\\:f`` no repetidas."""
function metodo_newton(f,x0,repeat="no-repeat")
    
    x = x0
    
    for i in 1:20
        x -= f(x)./derivada_numerica(f,x)
    end
    
    if repeat == "repeat"
        return x
    end
    
    return union(x)
    
end

"""Aproxima la integral de una función ``\\:f`` sobre ``[a,b]`` utilizando el método del rectángulo, `metodo_rectangulo```(\\:f,a,b,n)`` requiere de 4 argumentos: la función ``\\:f``, los dos extremos del intervalo de integración ``a``, ``b`` y el número de subintervalos ``n`` en los que se dividide el intervalo de integración."""
function metodo_rectangulo(f,a,b,n)
    
    R = 0
    h = (b-a)/n
    
    for i in 1:n
        R += h*f(a+(i-1/2)*h)
    end
    
    return R
    
end

"""Aproxima la integral de una función ``\\:f`` sobre ``[a,b]`` utilizando el método del trapecio, `metodo_trapecio```(\\:f,a,b,n)`` requiere de 4 argumentos: la función ``\\:f``, los dos extremos del intervalo de integración ``a``, ``b`` y el número de subintervalos ``n`` en los que se dividide el intervalo de integración."""
function metodo_trapecio(f,a,b,n)
    
    T = 0
    h = (b-a)/n
    
    for i in 1:n
        T += (h/2)*(f(a+i*h-h)+f(a+i*h))
    end
    
    return T
    
end

"""Aproxima la integral de una función ``\\:f`` sobre ``[a,b]`` utilizando el método de Simpson, `metodo_simpson```(\\:f,a,b,n)`` requiere de 4 argumentos: la función ``\\:f``, los dos extremos del intervalo de integración ``a``, ``b`` y el número de subintervalos ``n`` en los que se dividide el intervalo de integración."""
function metodo_simpson(f,a,b,n)
    
    S = 0
    h = (b-a)/n
    
    for i in 1:n
        S += (h/6)*(f(a+i*h-h)+4*f(a+i*h-h/2)+f(a+i*h))
    end
    
    return S
    
end

"""Dados dos vectores ``\\mathbf{x}`` y ``\\mathbf{y}`` en ``\\mathbb{R}^{n}`` (con componentes ``x_{k}`` y ``y_{k}``, respectivamente), `interpolador_lagrange```(\\mathbf{x},\\mathbf{y},x_{0})`` regresa el valor del polinomio de Lagrange ``L`` que pasa por los puntos ``(x_{k},y_{k})\\:\\forall k=1,\\dots,n`` evaluado en ``x_{0}``, i.e. nos da ``L(x_{0})``. Si los vectores ``\\mathbf{x}`` y ``\\mathbf{y}`` no son de la misma dimensión la función advierte la inconsistencia."""
function interpolador_lagrange(x,y,x0)
    
    lx = length(x)
    ly = length(y)
    if lx != ly
        return latexstring("Los vectores \$\\textbf{no}\$ son de la misma dimensión, \$\\mathbf{x}\\in\\mathbb{R}^{$lx}\$ y \$\\mathbf{y}\\in\\mathbb{R}^{$ly}\$.")
    end
    
    l = []
    
    for j in 1:lx
        z = 1
        for m in 1:lx
            if m != j
                z *= (x0-x[m])/(x[j]-x[m])
            end
        end
        push!(l,z)
    end
    
    return vecdot(y,l)
    
end

"""La función `euler` utiliza el algoritmo del método de Euler para aproximar la solución del sistema de ecuaciones diferenciales:

``\\dfrac{d\\mathbf{x}}{dt}=\\boldsymbol{f}(\\mathbf{x},t)``, con condiciones iniciales ``\\mathbf{x}(t_{0})=\\mathbf{x}_{0}``.

Así `euler```(\\:\\boldsymbol{f},t,\\mathbf{x}_{0})`` aproxima a ``\\mathbf{x}`` en ``t>t_{0}`` (donde ``t`` es una lista de varios tiempos separados ``h``). El método de Euler es:

``\\mathbf{x}_{k+1}=\\mathbf{x}_{k}+h\\:\\boldsymbol{f}(\\mathbf{x}_{k},t_{k})``,

donde ``\\mathbf{x}_{k}`` es la aproximación de ``\\mathbf{x}(t_{k})`` y ``t_{k}=t_{0}+hk``. Si se tiene ``\\mathbf{x}(t)=(x_{1}(t),\\dots,x_{n}(t))`` con ``n>1``, entonces `euler```(\\:\\boldsymbol{f},t,\\mathbf{x}_{0})`` `[i]` es el vector ``(x_{i}(t_{0}),\\dots,x_{i}(t_{m}))``, con ``m=`` `length```(t)``; si ``n=1`` (i.e. ``\\mathbf{x}(t)\\longrightarrow x(t)``, ``\\:\\boldsymbol{f}\\longrightarrow\\:f`` ), entonces `euler```(\\:f,t,x_{0})`` es directamente el vector ``(x(t_{0}),\\dots,x(t_{m}))``."""
function euler(f,listt,x0)
    
    x = x0
    h = listt[2]-listt[1]
    listx = []
    push!(listx,x)
    E = []
    
    for i in 2:length(listt)
        t = listt[i-1]
        x += h*f(x,t)
        push!(listx,x) 
    end
    
    if length(x0) == 1
        return listx
    end
    
    for j in 1:length(x0)
        push!(E,map(x->x[j],listx))
    end
    
    return E
    
end

"""La función `euler_implicito` utiliza el algoritmo del método de Euler implícito para aproximar la solución del sistema de ecuaciones diferenciales:

``\\dfrac{d\\mathbf{x}}{dt}=\\boldsymbol{f}(\\mathbf{x},t)``, con condiciones iniciales ``\\mathbf{x}(t_{0})=\\mathbf{x}_{0}``.

Así `euler_implicito```(\\:\\boldsymbol{f},t,\\mathbf{x}_{0}``,`"metodo"`) aproxima a ``\\mathbf{x}`` en ``t>t_{0}`` (donde ``t`` es una lista de varios tiempos separados ``h``). El método de Euler implícito es:

``\\mathbf{x}_{k+1}=\\mathbf{x}_{k}+h\\:\\boldsymbol{f}(\\mathbf{x}_{k+1},t_{k+1})``,

donde ``\\mathbf{x}_{k}`` es la aproximación de ``\\mathbf{x}(t_{k})`` y ``t_{k}=t_{0}+hk``. Para invertir el sistema algebraico anterior, `euler_implicito` cuenta con dos métodos a elegir en el cuarto argumento de la función (deben de escribirse dentro de las comillas dobles):

* `metodo`=`newton`: Utiliza el método de Newton para encontrar ``\\mathbf{x}_{k+1}``.
* `metodo`=`fijo`: Utiliza el método del punto fijo para encontrar ``\\mathbf{x}_{k+1}``

El cuarto argumento se puede omitir, en tal caso el método utilizado por defecto es el método de Newton. Si se tiene ``\\mathbf{x}(t)=(x_{1}(t),\\dots,x_{n}(t))``, entonces `euler_implicito```(\\:\\boldsymbol{f},t,\\mathbf{x}_{0}``,`"metodo"`) `[i]` es el vector ``(x_{i}(t_{0}),\\dots,x_{i}(t_{m}))``, con ``m=`` `length```(t)``; si ``n=1`` (i.e. ``\\mathbf{x}(t)\\longrightarrow x(t)``, ``\\:\\boldsymbol{f}\\longrightarrow\\:f`` ), entonces `euler_implicito```(\\:f,t,x_{0}``,`"metodo"`) es directamente el vector ``(x(t_{0}),\\dots,x(t_{m}))``."""
function euler_implicito(f,listt,x0,metodo="newton")
    
    h = listt[2]-listt[1]
    listx = []
    push!(listx,x0)
    EI = []
    
    for k in 2:length(listt)
        xk = listx[k-1]
        t = listt[k]
        if metodo == "newton"
            g(z) = z-xk-h*f(z,t)
            push!(listx,metodo_newton(g,xk,"repeat"))
        elseif metodo == "fijo"
            G(z) = xk+h*f(z,t)
            push!(listx,punto_fijo(G,xk))
        else
            return print("El método debe de ser \x1b[1mnewton\x1b[0m o \x1b[1mfijo\x1b[0m, no está definido para \x1b[1m$metodo\x1b[0m")
        end
    end
    
    if length(x0) == 1
        return listx
    end
    
    for j in 1:length(x0)
        push!(EI,map(x->x[j],listx))
    end
    
    return EI
    
end

"""La función `RK4` utiliza el algoritmo del método de Runge-Kutta de orden 4 (método RK4) para aproximar la solución del sistema de ecuaciones diferenciales:

``\\dfrac{d\\mathbf{x}}{dt}=\\boldsymbol{f}(\\mathbf{x},t)``, con condiciones iniciales ``\\mathbf{x}(t_{0})=\\mathbf{x}_{0}``.

Así `RK4```(\\:\\boldsymbol{f},t,\\mathbf{x}_{0})`` aproxima a ``\\mathbf{x}`` en ``t>t_{0}`` (donde ``t`` es una lista de varios tiempos separados ``h``). El método RK4 es:

``\\mathbf{x}_{i+1}=\\mathbf{x}_{i}+\\dfrac{1}{6}h(\\mathbf{k}_{1}+2\\mathbf{k}_{2}+2\\mathbf{k}_{3}+\\mathbf{k}_{4})``, donde:

``\\mathbf{k}_{1}=\\boldsymbol{f}\\left(\\mathbf{x}_{i},t_{i}\\right)``

``\\mathbf{k}_{2}=\\boldsymbol{f}\\left(\\mathbf{x}_{i}+\\dfrac{1}{2}h\\mathbf{k}_{1},t_{i}+\\dfrac{1}{2}h\\right)``

``\\mathbf{k}_{3}=\\boldsymbol{f}\\left(\\mathbf{x}_{i}+\\dfrac{1}{2}h\\mathbf{k}_{2},t_{i}+\\dfrac{1}{2}h\\right)``

``\\mathbf{k}_{4}=\\boldsymbol{f}\\left(\\mathbf{x}_{i}+h\\mathbf{k}_{3},t_{i}+h\\right)``

y con ``\\mathbf{x}_{k}`` la aproximación de ``\\mathbf{x}(t_{k})`` y ``t_{k}=t_{0}+hk``. Si se tiene ``\\mathbf{x}(t)=(x_{1}(t),\\dots,x_{n}(t))``, entonces `RK4```(\\:\\boldsymbol{f},t,\\mathbf{x}_{0})`` `[i]` es el vector ``(x_{i}(t_{0}),\\dots,x_{i}(t_{m}))``, con ``m=`` `length```(t)``; si ``n=1`` (i.e. ``\\mathbf{x}(t)\\longrightarrow x(t)``, ``\\:\\boldsymbol{f}\\longrightarrow\\:f`` ), entonces `RK4```(\\:f,t,x_{0})`` es directamente el vector ``(x(t_{0}),\\dots,x(t_{m}))``."""
function RK4(f,listt,x0)
    
    x = x0
    l = (listt[2]-listt[1])/2.0
    listx = []
    push!(listx,x)
    RK = []
    
    for i in 2:length(listt)
        t = listt[i-1]
        k1 = f(x,t)
        k2 = f(x+l*k1,t+l)
        k3 = f(x+l*k2,t+l)
        k4 = f(x+2l*k3,t+2l)
        x += (l/3.0)*(k1+2k2+2k3+k4)
        push!(listx,x)
    end
    
    if length(x0) == 1
        return listx
    end
    
    for j in 1:length(x0)
        push!(RK,map(x->x[j],listx))
    end
    
    return RK
    
end

end