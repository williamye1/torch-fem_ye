from abc import ABC, abstractmethod
from math import sqrt
from typing import Callable

import torch
from torch import Tensor

from torchfem.utils import (
    stiffness2voigt,
    strain2voigt,
    stress2voigt,
    voigt2stiffness,
    voigt2stress,
)


class Material(ABC):
    """Base class for material models."""

    @abstractmethod
    def __init__(self):
        self.n_state: int
        self.is_vectorized: bool
        self.C: Tensor
        pass

    @abstractmethod
    def vectorize(self, n_elem: int):
        pass

    @abstractmethod
    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        sigma: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def rotate(self, R: Tensor):
        pass


class IsotropicElasticity3D(Material):
    """Isotropic elastic material.

    This class represents a 3D isotropic linear elastic material under small-strain
    assumptions, defined by Young's modulus E and Poisson's ratio ν.

    Attributes:
        E (Tensor): Young's modulus. If a float is provided, it is converted.
            Shape: `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor): Poisson's ratio. If a float is provided, it is converted.
            Shape: `()` for a scalar or `(N,)` for a batch of materials.
        n_state (int): Number of internal state variables (here: 0).
        is_vectorized (bool): `True` if `E` and `nu` have batch dimensions.
        lbd (Tensor): First Lamé parameter.
            Shape: `()` (scalar) or `(N,)` (batch).
        G (Tensor): Shear modulus (second Lamé parameter).
            Shape: `()` (scalar) or `(N,)` (batch).
        C (Tensor): Fourth-order elasticity tensor for 3D isotropic elasticity.
            Shape: `(N, 3, 3, 3, 3)` if vectorized, otherwise `(3, 3, 3, 3)`.
    """

    def __init__(self, E: float | Tensor, nu: float | Tensor):
        # Convert float inputs to tensors
        if isinstance(E, float):
            E = torch.tensor(E)
        if isinstance(nu, float):
            nu = torch.tensor(nu)

        # Store material properties
        self.E = E
        self.nu = nu

        # There are no internal variables
        self.n_state = 0

        # Check if the material is vectorized
        self.is_vectorized = E.dim() > 0

        # Lame parameters
        self.lbd = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.G = self.E / (2.0 * (1.0 + self.nu))

        # Identity tensors
        I2 = torch.eye(3)
        I4 = torch.einsum("ij,kl->ijkl", I2, I2)
        I4S = torch.einsum("ik,jl->ijkl", I2, I2) + torch.einsum("il,jk->ijkl", I2, I2)

        # Stiffness tensor
        lbd = self.lbd[..., None, None, None, None]
        G = self.G[..., None, None, None, None]
        self.C = lbd * I4 + G * I4S

    def vectorize(self, n_elem: int):
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicElasticity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            return IsotropicElasticity3D(E, nu)

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        sigma: Tensor,
        state: Tensor,
        de0: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Performs an incremental step in the small-strain isotropic elasticity model.

        This function updates the deformation gradient, stress, and internal state
        variables based on a small-strain assumption.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 3, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 3, 3)`, same as `H_inc`.
            sigma (Tensor): Current Cauchy stress tensor.
                - Shape: `(..., 3, 3)`.
            state (Tensor): Internal state variables (unused in linear elasticity).
                - Shape: Arbitrary, remains unchanged.
            de0 (Tensor): External small strain increment (e.g., thermal).
                - Shape: `(..., 3, 3)`.

        Returns:
            tuple:
                - **sigma_new (Tensor)**: Updated Cauchy stress tensor.
                Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state (unchanged).
                Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                Shape: `(..., 3, 3, 3, 3)`.
        """
        # Compute small strain tensor
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        # Compute new stress
        sigma_new = sigma + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)
        # Update internal state (this material does not change state)
        state_new = state
        # Algorithmic tangent
        ddsdde = self.C
        return sigma_new, state_new, ddsdde

    def rotate(self, R: Tensor):
        """Rotates the material using a given rotation matrix.

        For isotropic materials, rotation has no effect, since their properties
        remain unchanged under coordinate transformations. This function exists for API
        consistency with anisotropic materials.

        Args:
            R (Tensor): Rotation matrix of shape `(3, 3)` or `(..., 3, 3)` for batched
            rotations.

        Returns:
            IsotropicElasticity3D: The same material instance (no effect).

        """
        print("Rotating an isotropic material has no effect.")
        return self
    

class IsotropicDamage3D(IsotropicElasticity3D):
    """Isotropic elastoplastic material model.

    This class extends `IsotropicElasticity3D` to incorporate isotropic plasticity
    with a von Mises yield criterion. The model follows a return-mapping algorithm
    for small strains and enforces associative plastic flow with a given yield function
    and its derivative.

    Attributes:
        E (Tensor): Young's modulus. If a float is provided, it is converted.
            Shape: `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor): Poisson's ratio. If a float is provided, it is converted.
            Shape: `()` for a scalar or `(N,)` for a batch of materials.
        n_state (int): Number of internal state variables (here: 1).
        is_vectorized (bool): `True` if `E` and `nu` have batch dimensions.
        sigma_f (Callable): Function that defines the yield stress as a function
            of the equivalent plastic strain.
        sigma_f_prime (Callable): Derivative of the yield function with respect to
            the equivalent plastic strain.
        tolerance (float, optional): Convergence tolerance for the plasticity
            return-mapping algorithm. Default is `1e-5`.
        max_iter (int, optional): Maximum number of iterations for the local Newton
            solver in plasticity correction. Default is `10`.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        # eps_0: float,
        # eps_f: float,
        d_kappa: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
    ):
        super().__init__(E, nu)
        self.d_kappa=d_kappa
        # self.eps_0=eps_0
        # self.eps_f=eps_f
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int):
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            return IsotropicDamage3D(
                E, nu, self.d_kappa,self.tolerance, self.max_iter
            )

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        sigma: Tensor,
        state: Tensor,
        de0: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:     
        """Perform a strain increment with an elastoplastic model using small strains.

        This function updates the deformation gradient, computes the small strain
        tensor, evaluates trial stress, and updates the stress based on the yield
        condition and flow rule. The algorithm uses a local Newton solver to find the
        plastic strain increment and adjusts the stress and internal state accordingly.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 3, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 3, 3)`, same as `H_inc`.
            sigma (Tensor): Current Cauchy stress tensor.
                - Shape: `(..., 3, 3)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain.
                - Shape: `(..., 1)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                - Shape: `(..., 3, 3)`.

        Returns:
            tuple:
                - **sigma_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 3, 3, 3, 3)`.
        """
        # Second order identity tensor
        #I2 = torch.eye(F.shape(-1))

        # Compute small strain tensor

        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)

        I = torch.eye(3, device=F.device).expand_as(F)
        H=F-I
        epsilon = 0.5 * (H.transpose(-1, -2) + H)

        # Initialize solution variables
        sigma_new = sigma.clone()
        state_new = state.clone()
        kappa=state_new.clone()[...,0]
        #ddsdde = self.C.clone()

        eps_trial=epsilon+de

        s_trial = torch.einsum("...ijkl,...kl->...ij", self.C, eps_trial - de0)

        #Calculate equivalent strain 
        eps_trial_trace = eps_trial[..., 0, 0] + eps_trial[..., 1, 1] + eps_trial[..., 2, 2]
        dev_eps_trial=eps_trial.clone()
        dev_eps_trial[..., 0, 0] -= eps_trial_trace / 3
        dev_eps_trial[..., 1, 1] -= eps_trial_trace / 3
        dev_eps_trial[..., 2, 2] -= eps_trial_trace / 3

        # #Compute Von Mises equivalent strain
        eps_eq= torch.sqrt(2/3 * torch.sum(dev_eps_trial * dev_eps_trial, dim=(-1, -2)))

        #Update kappa
        kappa=torch.maximum(kappa,eps_eq)  

        ###fonction f est implicitee dedans car si kappa_precdent<eps_eq, on charge
        #f>0 donc les valeurs de dommage augmentent et si kappa_precedent>eps_eq alors f<0
        #donc on decharge donc pas d'augmententation d'endommagement

        #Calculate damage
        D = self.d_kappa(kappa)

        print(D.max())
        
    
        assert torch.all(D<=1), "Damage parameter>1"

        if torch.allclose(F[0], torch.eye(F[0].shape[0])) and D.max() > 0:
            raise Exception("First step must be lower than epsilon0. Reduce increment.")

        damage=(1 - D[:,None,None])


        #update sigma
        sigma_new=damage*s_trial


        #print("epsilon")
        #print(eps_eq[0])
        # print("D=")
        #print(D)
        #sigma_new=damage*sigma

        #update state
        state_new[..., 0] = kappa

        
        ddsdde=(1.0-D)[:, None, None,None,None] *self.C.clone()
        #print(ddsdde)


        return sigma_new, state_new, ddsdde




