from typing import Callable, List, Sequence, Tuple, Union

import gsw
import numpy as np


def fp(
    d: float,
    S: float,
    λ_1: float = -0.0573,
    λ_2: float = 0.0832,
    λ_3: float = 0.000761,
) -> float:
    """Linearized equation for the freezing point temperature for seawater.

    Parameters
    ----------
        d : float
            Depth (m).
        S : float
            Salinity (PSU).
        λ_1 : float, optional
            Variation of freezing point with salinity, default -0.0573 (C PSU-1).
        λ_2 : float, optional
            Freezing point offset, default 0.0832 (C).
        λ_3 : float, optional
            Variation of freezing point with depth, default 0.000761 (C m-1).

    Returns
    -------
        T : float
            Freezing point temperature (C).

    """
    return λ_1 * S + λ_2 + λ_3 * d


def melt_rate(
    d: float,
    T: float,
    S: float,
    u: float,
    Γ_T: float = 0.022,
    Γ_S: float = 0.00062,
    C_d: float = 0.0025,
    T_i: float = -10.0,
    λ_1: float = -0.0573,
    λ_2: float = 0.0832,
    λ_3: float = 0.000761,
    L: float = 335000.0,
    c_i: float = 2009.0,
    c_w: float = 3974.0,
) -> Tuple[float, float, float]:
    """The 3-equation melt rate parameterization (e.g. Holland and Jenkins 1999).

    Parameters
    ----------
        d : float
            Depth (m).
        T : float
            Outer boundary layer temperature (C).
        S : float
            Outer boundary layer salinity (PSU).
        u : float
            Outer boundary layer speed (m s-1).
        Γ_T : float, optional
            Turbulent transfer coefficient for temperature, default 0.022.
        Γ_S : float, optional
            Turbulent transfer coefficient for salinity, default 0.00062.
        C_d : float, optional
            Drag coefficient, default 0.0025.
        T_i : float, optional
            Ice temperature, default -10.0 (C).
        λ_1 : float, optional
            Variation of freezing point with salinity in linear freezing point
            of seawater, default -0.0573 (C PSU-1).
        λ_2 : float, optional
            Freezing point offset in linear freezing point of seawater,
            default 0.0832 (C).
        λ_3 : float, optional
            Variation of freezing point with depth in linear freezing point of
            seawater, default 0.000761 (C m-1).
        L : float, optional
            Latent heat of fusion, default 335000.0 (J kg-1).
        c_i : float, optional
            Heat capacity of ice, default 2009.0 (J kg-1 C-1).
        c_w : float, optional
            heat capacity of seawater, default 3974.0 (J kg-1 C-1).

    Returns
    -------
        m : float
            Melt rate (m s-1).
        T_b : float
            Ice-ocean interface temperature (C).
        S_b : float
            Ice-ocean interface salinity (PSU).
    """
    T_b = fp(d, S, λ_1, λ_2, λ_3)
    # Coefficients of the quadratic equation for interface salinity
    a = λ_1 * (Γ_T * c_w - Γ_S * c_i)
    b = Γ_S * c_i * (λ_1 * S - λ_2 - λ_3 * d + T_i - (L / c_i)) - Γ_T * c_w * (
        T - λ_2 - λ_3 * d
    )
    c = Γ_S * S * (c_i * (λ_2 + λ_3 * d - T_i) + L)
    # Salinity of the ice-ocean interface (PSU)
    S_b = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
    m = Γ_S * C_d**0.5 * u * (S - S_b) / S_b  # Melt rate (m s-1)
    return m, T_b, S_b


def bpt(
    z: float,
    fluxes: Sequence[float],
    T: Union[float, Callable[[float], float]],
    S: Union[float, Callable[[float], float]],
    u: Union[float, Callable[[float], float]] = 0.0,
    α: float = 0.1,
    ρ_0: float = 1025.0,
    g: float = -9.81,
    lat: float = 60.0,
    lon: float = -130.0,
    geometry: str = "line",
    Γ_T: float = 0.022,
    Γ_S: float = 0.00062,
    C_d: float = 0.0025,
    **melt_kwargs,
) -> List[int]:
    """Differential equations of buoyant plume theory for a line plume with ice melt.

    Parameters
    ----------
        z : float
            Height or negative depth (m).
        fluxes : array_like of float
            Plume fluxes of [mass, momentum, temperature, salt].
        T : callable or float
            Ocean (far-field) in-situ temperature profile as a function of
            height, T(z) (C), or a constant value. Default 5.
        S : callable or float
            Ocean (far-field) practical salinity profile as a function of
            height, S(z) (PSU), or a constant value. Default 25.
        u : callable or float
            Additional horizontal ocean velocity as a function of height,
            u(z) (m s-1), or a constant value. Default 0.
        α : float, optional
            Entrainment coefficient (m s-1).
        ρ_0 : float, optional
            Density constant, default 1025.0 (kg m-3).
        g : float, optional
            Gravitational constant, default -9.81 (m s-2).
        lat : float, optional
            Latitude used in the pressure calculation, default 60 (degree_north).
        lon : float, optional
            Longitude used in the absolute salinity calculation, default -130 (degree_east).
        geometry : str, optional
            Initialize plume with a line (default) or point geometry.
        Γ_T : float, optional
            Turbulent transfer coefficient for temperature, default 0.022 .
        Γ_S : float, optional
            Turbulent transfer coefficient for salinity, default 0.00062.
        C_d : float, optional
            Drag coefficient, default 0.0025.
        melt_kwargs : dict, optional
            Other optional parameters for the melt rate calculation.

    Returns
    -------
        dfluxes : list
            Vertical derivative of plume fluxes of [mass, momentum, temperature, salt]
    """
    # Current state
    mass_flux = fluxes[0]  # Mass flux (m2 s-1)
    mom_flux = fluxes[1]  # Momentum flux (m3 s-2)
    T_flux = fluxes[2]  # Temperature flux (C m2 s-1)
    S_flux = fluxes[3]  # Salt flux (PSU m2 s-1)

    w = mom_flux / mass_flux  # Plume vertical velocity (m s-1)
    R = mass_flux / w  # Plume thickness or radius (m)
    T_p = T_flux / mass_flux  # Plume temperature (C)
    S_p = S_flux / mass_flux  # Plume salinity (PSU)

    if callable(S):
        S_o = S(z)
    else:
        S_o = S

    if callable(T):
        T_o = T(z)
    else:
        T_o = T

    if callable(u):
        u_o = u(z)
    else:
        u_o = u

    # Velocity relevat to the melt rate calculation
    U = (u_o**2 + w**2) ** 0.5

    # Parameters for the ODEs
    m, T_b, S_b = melt_rate(-z, T_p, S_p, U, Γ_T=Γ_T, Γ_S=Γ_S, C_d=C_d, **melt_kwargs)
    p = gsw.p_from_z(z, lat)  # Pressure (dbar)
    SA_p = gsw.SA_from_SP(S_p, p, lon, lat) # Plume absolute salinity
    SA_o = gsw.SA_from_SP(S_o, p, lon, lat) # Ocean absolute salinity
    ρ_o = gsw.pot_rho_t_exact(SA_p, T_p, p, p_ref=0) # Plume potential density (kg m-3)
    ρ = gsw.pot_rho_t_exact(SA_o, T_o, p, p_ref=0) # Ocean (far-field) potential density (kg m-3)
    gp = g * (ρ_o - ρ) / ρ_0  # Reduced gravity (m s-2)

    dfluxes = [0, 0, 0, 0]
    if geometry == "line":
        dfluxes[0] = α * w + m  # Mass
        dfluxes[1] = R * gp - C_d * w**2  # Momentum
        dfluxes[2] = α * w * T_o + m * T_b - C_d**0.5 * U * Γ_T * (T_p - T_b)  # Temperature
        dfluxes[3] = α * w * S_o + m * S_b - C_d**0.5 * U * Γ_S * (S_p - S_b)  # Salinity
    elif geometry == "point":
        dfluxes[0] = 2 * α * R * w + 4 * R * m / np.pi  # Mass
        dfluxes[1] = R**2 * gp - 4 * C_d * R * w**2 / np.pi  # Momentum
        dfluxes[2] = 2 * α * R * w * T_o + 4 * m * R * T_b / np.pi - 4 * C_d**0.5 * R * U * Γ_T * (T_p - T_b) / np.pi  # Temperature
        dfluxes[3] = 2 * α * R * w * S_o + 4 * m * R * S_b / np.pi - 4 * C_d**0.5 * R * U * Γ_S * (S_p - S_b) / np.pi  # Salinity
    else:
        raise ValueError(f"geometry must be 'point' or 'line', got {geometry!r}")
    return dfluxes


def Δρ(
    z: float,
    fluxes: Sequence[float],
    T: Union[float, Callable[[float], float]],
    S: Union[float, Callable[[float], float]],
    u: Union[float, Callable[[float], float]] = 0.0,
    α: float = 0.1,
    ρ_0: float = 1025.0,
    g: float = -9.81,
    lat: float = 60.0,
    lon: float = -130.0,
    geometry: str = "line", 
) -> float:
    """Density difference between plume and ocean."""
    T_p = fluxes[2] / fluxes[0]  # Plume temperature (C)
    S_p = fluxes[3] / fluxes[0]  # Plume salinity (PSU)

    if callable(T) and callable(S):
        T_o = T(z)
        S_o = S(z)
    else:
        T_o = T
        S_o = S

    p = gsw.p_from_z(z, lat)  # Pressure (dbar)
    SA_p = gsw.SA_from_SP(S_p, p, lon, lat) # Plume absolute salinity
    SA_o = gsw.SA_from_SP(S_o, p, lon, lat) # Ocean absolute salinity
    ρ_o: float = gsw.pot_rho_t_exact(SA_p, T_p, p, p_ref=0)  # Plume potential density (kg m-3)
    ρ: float = gsw.pot_rho_t_exact(SA_o, T_o, p, p_ref=0)  # Ocean (far-field) potential density (kg m-3)
    return ρ - ρ_o



# These attributes are used by scipy.integrate.solve_ivp to terminate integration
Δρ.terminal = True
Δρ.direction = -1
