# Option 2: Updates in the Source and Software Architecture

## Matplotlib’s Overall Architecture and How Text Rendering Fits In

![architecture](architecture.svg)

We draw a simple flowchart which shows the matplotlib architecture, matplotlib is a comprehensive plotting library that follows a modular architecture. Its core components include:

1. **Figure and Axes Layer (Frontend API)**
   - Users interact with `Figure` and `Axes` objects through functions like `plt.plot()`, `ax.text()`, etc.
   - This layer provides a high-level abstraction for plotting and annotation placement.

2. **Artist Layer (Core Drawing Primitives)**
   - `Text`, `Line2D`, `Patch`, and other primitives define graphical elements.
   - `Text` objects, which we modified, reside in this layer and handle string rendering.

3. **Transformation Layer (Coordinate Mapping and Layout)**
   - This layer computes layout positioning, transforms text and graphical elements between user space and display space.
   - `_get_layout()` in `text.py` plays a critical role here.

4. **Backend Layer (Rendering and Output Generation)**
   - Matplotlib supports multiple backends (`Agg`, `SVG`, `PDF`, etc.), which handle rendering.
   - `draw_text()` and `draw_mathtext()` delegate the actual text drawing to these backends.

5. **MathText System (`mathtext.py`, `_mathtext.py`)**
   - MathText parsing and rendering occur here, allowing LaTeX-style expressions inside text elements.
   - The `MathTextParser` converts math expressions into graphical representations.

This layered design ensures that Matplotlib can handle diverse plotting needs while maintaining flexibility in rendering across different backends.

Note: We draw the matplotlib achitecture diagram using [Mermaid Live Editor](https://www.mermaidchart.com/play)

---

## How Our Modifications Impact Matplotlib’s Architecture

### **1. Disrupting the Flexible Rendering Pipeline**
- By introducing `exist_math`, we altered the way Matplotlib dynamically chooses rendering methods.
- Instead of allowing mixed text to be processed flexibly, our change forced a global decision that affected all text elements within a `Text` object.
- **Impact:** This rigid approach reduced Matplotlib’s ability to adapt text rendering based on different contexts and backends.

### **2. Affecting Multi-Backend Support**
- Matplotlib is designed to support multiple backends (`Agg`, `SVG`, `PDF`, etc.), each with different text rendering capabilities.
- Our changes in `text.py` influenced text layout calculations that should ideally remain backend-independent.
- **Impact:** By modifying `draw()` and `_get_layout()` in a hardcoded way, we risked breaking rendering consistency across backends.

### **3. Breaking the Separation Between Text and MathText**
- The MathText system (`mathtext.py`) is designed to handle LaTeX-style expressions separately from standard text.
- Our approach forced all text containing math expressions into the MathText rendering path, which is not how Matplotlib was originally designed.
- **Impact:** This violated Matplotlib’s modularity, making text rendering logic less clean and harder to maintain.

---

## A More Architecturally Sound Solution

Our solution is not a solid one for this issue, while it has negative effect on the original matplotlib architecture. The refactoring leads to failure in the original tests, thus, we need a more solid one and we propose possibles ways which could be used as future work.

### **1. Refactoring MathText Alignment Within the MathText System**
- Instead of modifying `text.py`, we should enhance `mathtext.py` to ensure consistent bounding box computation for MathText and normal text.
- The correct approach involves refining `MathTextParser` to adjust MathText spacing dynamically rather than forcing an `exist_math` flag.

```python
class MathTextAdapter:
    def __init__(self, math_bbox):
        self.math_bbox = math_bbox

    def align_with_normal_text(self, normal_bbox):
        kerning_offset = abs(self.math_bbox.x1 - normal_bbox.x1)
        self.math_bbox.apply_offset(kerning_offset)
```

### **2. Preserving Multi-Backend Compatibility**
- Instead of hardcoding changes in `draw()`, we should adjust text layout computations in `_get_layout()`.
- The fix should be applied at the transformation layer, ensuring that different backends can still process text elements correctly.

```python
def _get_layout(self):
    if self.contains_math():
        math_bbox = self._mathtext_parser.get_bbox(...)
        normal_bbox = self._get_normal_text_bbox(...)
        
        adapter = MathTextAdapter(math_bbox)
        adapter.align_with_normal_text(normal_bbox)
```

### **3. Using Strategy Pattern for Backend Rendering Decisions**
- Rather than introducing a global flag, we can refactor `Text` rendering to delegate backend decisions dynamically.
- **Example Strategy Pattern Implementation:**
  ```python
  class TextRenderer:
      def render(self, text_obj):
          if text_obj.contains_math():
              return MathTextRenderer().render(text_obj)
          else:
              return StandardTextRenderer().render(text_obj)
  ```

---

## Refactoring Principles and Design Patterns Used

### **Adapter Pattern (Aligning MathText Bounding Boxes with Normal Text)**
- MathText and normal text should align at the bounding box level without affecting overall rendering logic.

### **Strategy Pattern (Decoupling Backend Rendering Decisions)**
- Instead of hardcoded `exist_math`, backends dynamically select the appropriate rendering strategy.

### **Single Responsibility Principle (SRP) - Keeping Text and MathText Processing Separate**
- MathText adjustments should remain within `mathtext.py`, rather than modifying `text.py` globally.

---

## Conclusion
Our initial modifications unintentionally disrupted Matplotlib’s modular architecture by enforcing a rigid text rendering decision. A better approach ensures that:

- **MathText adjustments are handled in `mathtext.py`, not `text.py`**, preserving separation of concerns.
- **Multi-backend compatibility is maintained**, allowing rendering consistency across different output formats.
- **Text and MathText remain distinct entities**, with proper spacing adjustments at the layout computation level.
- **Design patterns like Adapter and Strategy are used**, making the solution more maintainable and extensible.

By addressing the issue within the MathText processing system rather than modifying `text.py`, we align our fix with Matplotlib’s overall architectural principles.
