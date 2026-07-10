Improved tick placement for ``symlog`` axes
-------------------------------------------

The placement of ticks for ``symlog`` axes has been improved. Ticks are now
placed identically to ``log`` axes in the logarithmic part with a reasonable
extension of this behavior to the linear part of the axis. Axes with too few
ticks or spurious ticks are avoided by the new implementation.
