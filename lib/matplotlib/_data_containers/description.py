from dataclasses import dataclass
from typing import TypeAlias, Union, overload


ShapeSpec: TypeAlias = tuple[Union[str, int], ...]


@dataclass(frozen=True)
class Desc:
    # TODO: sort out how to actually spell this.  We need to know:
    #   - what the number of dimensions is (1d vs 2d vs ...)
    #   - is this a fixed size dimension (e.g. 2 for xextent)
    #   - is this a variable size depending on the query (e.g. N)
    #   - what is the relative size to the other variable values (N vs N+1)
    # We are probably going to have to implement a DSL for this (😞)
    shape: ShapeSpec
    coordinates: str = "auto"

    @staticmethod
    def validate_shapes(
        specification: dict[str, ShapeSpec | "Desc"],
        actual: dict[str, ShapeSpec | "Desc"],
        *,
        broadcast: bool = False,
    ) -> None:
        """Validate specified shape relationships against a provided set of shapes.

        Shapes provided are tuples of int | str. If a specification calls for an int,
        the exact size is expected.
        If it is a str, it must be a single capital letter optionally followed by ``+``
        or ``-`` an integer value.
        The same letter used in the specification must represent the same value in all
        appearances. The value may, however, be a variable (with an offset) in the
        actual shapes (which does not need to have the same letter).

        Shapes may be provided as raw tuples or as ``Desc`` objects.

        Parameters
        ----------
        specification: dict[str, ShapeSpec | "Desc"]
           The desired shape relationships
        actual: dict[str, ShapeSpec | "Desc"]
           The shapes to test for compliance

        Keyword Parameters
        ------------------
        broadcast: bool
           Whether to allow broadcasted shapes to pass (i.e. actual shapes with a ``1``
           will not cause exceptions regardless of what the specified shape value is)

        Raises
        ------
        KeyError:
            If a required field from the specification is missing in the provided actual
            values.
        ValueError:
            If shapes are incompatible in any other way
        """
        specvars: dict[str, int | tuple[str, int]] = {}
        for fieldname in specification:
            spec = specification[fieldname]
            if fieldname not in actual:
                raise KeyError(
                    f"Actual is missing {fieldname!r}, required by specification."
                )
            desc = actual[fieldname]
            if isinstance(spec, Desc):
                spec = spec.shape
            if isinstance(desc, Desc):
                desc = desc.shape
            if not broadcast:
                if len(spec) != len(desc):
                    raise ValueError(
                        f"{fieldname!r} shape {desc} incompatible with specification "
                        f"{spec}."
                    )
            elif len(desc) > len(spec):
                raise ValueError(
                    f"{fieldname!r} shape {desc} incompatible with specification "
                    f"{spec}."
                )
            for speccomp, desccomp in zip(spec[::-1], desc[::-1]):
                if broadcast and desccomp == 1:
                    continue
                if isinstance(speccomp, str):
                    specv, specoff = speccomp[0], int(speccomp[1:] or 0)
                    entry: tuple[str, int] | int

                    if isinstance(desccomp, str):
                        descv, descoff = desccomp[0], int(desccomp[1:] or 0)
                        entry = (descv, descoff - specoff)
                    else:
                        entry = desccomp - specoff

                    if specv in specvars and entry != specvars[specv]:
                        raise ValueError(f"Found two incompatible values for {specv!r}")

                    specvars[specv] = entry
                elif speccomp != desccomp:
                    raise ValueError(
                        f"{fieldname!r} shape {desc} incompatible with specification "
                        f"{spec}"
                    )
        return None

    @staticmethod
    def compatible(
        a: dict[str, "Desc"],
        b: dict[str, "Desc"],
        aliases: tuple[tuple[str, str], ...] = (),
    ) -> bool:
        """Determine if ``a`` is a valid input for ``b``.

        Note: ``a`` _may_ have additional keys.
        """

        def resolve_aliases(coord):
            while True:
                for coa, cob in aliases:
                    if coord == coa:
                        coord = cob
                        break
                else:
                    break
            return coord

        try:
            Desc.validate_shapes(b, a)   # type: ignore[arg-type]
        except (KeyError, ValueError):
            return False
        for k, v in b.items():
            if resolve_aliases(a[k].coordinates) != resolve_aliases(v.coordinates):
                return False
        return True


@overload
def desc_like(desc: Desc, shape=None, coordinates=None) -> Desc: ...


@overload
def desc_like(
    desc: dict[str, Desc], shape=None, coordinates=None
) -> dict[str, Desc]: ...


def desc_like(desc, shape=None, coordinates=None):
    if isinstance(desc, dict):
        return {k: desc_like(v, shape, coordinates) for k, v in desc.items()}
    if shape is None:
        shape = desc.shape
    if coordinates is None:
        coordinates = desc.coordinates
    return Desc(shape, coordinates)


# Monkey patch mpl_data_containers for Desc isinstance checks
try:
    from mpl_data_containers import description
    description.Desc = Desc
except ImportError:
    pass
