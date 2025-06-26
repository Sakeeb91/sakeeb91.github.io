# Where to Find the Visuals in FitzHugh-Nagumo Lift & Learn

This guide shows you exactly where and when visualizations appear during execution.

## 🎯 **Quick Answer: Visuals Appear Automatically During Execution**

When you run the project, visualizations appear as **popup windows** during specific phases. Here's exactly when and where:

---

## 📍 **Visual Locations by Execution Phase**

### **Phase 1: Data Generation** 
```bash
python main.py --phase 1
```

**Visual Output:**
- **Sample Training Data Plot** (automatically appears)
  - **Location**: `main.py:110-125`
  - **Function**: `self.visualizer.plot_solution_evolution()`
  - **Shows**: Space-time evolution of first training parameter set
  - **File**: Popup window showing 2×2 subplot layout

---

### **Phase 3: POD Analysis**
```bash
python main.py --phase 3
```

**Visual Outputs:**
1. **POD Energy Spectrum** (automatically appears)
   - **Location**: `main.py:196`, calls `pod.plot_energy_spectrum()`
   - **File**: `src/pod_reduction.py:160-182`
   - **Shows**: Singular values and cumulative energy (2×1 layout)

2. **POD Mode Shapes** (automatically appears)
   - **Location**: `main.py:197`, calls `pod.plot_pod_modes()`
   - **File**: `src/pod_reduction.py:184-220`
   - **Shows**: Spatial structure of first 6 POD modes

---

### **Phase 5: Validation**
```bash
python main.py --phase 5
```

**Visual Outputs:**
1. **ROM vs High-Fidelity Comparison** (automatically appears)
   - **Location**: `main.py:293`, parameter `plot=True`
   - **Function**: `validator.validate_single_prediction()`
   - **File**: `src/validation.py:100-140` → calls `_plot_comparison()`
   - **Shows**: Detailed 3×3 comparison grid

2. **Error vs Dimension Plot** (if enabled)
   - **Location**: `main.py:335`
   - **Function**: `validator.plot_error_vs_dimension()`
   - **File**: `src/validation.py:436-459`
   - **Shows**: Key paper reproduction plot

---

### **Extension: Noise Analysis**
```bash
python main.py --phase extension
```

**Visual Outputs:**
1. **Noise Robustness Analysis** (automatically appears)
   - **Location**: `main.py:394`
   - **Function**: `noise_analyzer.plot_noise_analysis_results()`
   - **File**: `src/noise_extension.py:336-422`
   - **Shows**: 2×2 analysis of regularization performance

2. **Regularization Comparison** (automatically appears)
   - **Location**: `main.py:411`
   - **Function**: `noise_analyzer.plot_regularization_comparison()`
   - **File**: `src/noise_extension.py:627-662`
   - **Shows**: Hyperparameter optimization results

---

## 🖥️ **How Visuals Actually Appear**

### **Popup Windows**
- All plots open in **separate popup windows**
- **matplotlib backend**: Uses your system's default (TkAgg, Qt5Agg, etc.)
- **Interactive**: You can zoom, pan, save each plot
- **Blocking**: Program waits for you to close each plot before continuing

### **Example Execution Flow:**
```bash
cd "FitzHugh-Nagumo System"
source venv/bin/activate
python main.py  # Run all phases

# What happens:
# 1. Phase 1 runs → popup with training data plot appears
# 2. Close plot → Phase 2 runs (no visuals)  
# 3. Phase 3 runs → popup with POD energy plot appears
# 4. Close plot → popup with POD modes plot appears
# 5. Close plot → Phase 4 runs (no visuals)
# 6. Phase 5 runs → popup with ROM validation plot appears
# 7. Close plot → Extension runs → popup with noise analysis appears
# 8. And so on...
```

---

## 🎛️ **Controlling Visual Output**

### **Turn Off All Visuals**
Create a config file `no_visuals.json`:
```json
{
  "visualization": {
    "show_training_data": false,
    "show_pod_analysis": false,
    "show_validation": false,
    "show_dimension_study": false,
    "show_noise_analysis": false,
    "show_regularization_comparison": false
  }
}
```

Then run:
```bash
python main.py --config no_visuals.json
```

### **Enable Only Specific Visuals**
```json
{
  "visualization": {
    "show_training_data": false,
    "show_pod_analysis": true,
    "show_validation": true,
    "show_dimension_study": false,
    "show_noise_analysis": false,
    "show_regularization_comparison": false
  }
}
```

---

## 📊 **Complete List of Available Visuals**

### **Built into Main Pipeline:**

| Visual | Phase | Function | Description |
|--------|-------|----------|-------------|
| Training Data | 1 | `plot_solution_evolution()` | Space-time dynamics |
| POD Energy | 3 | `plot_energy_spectrum()` | Singular value analysis |
| POD Modes | 3 | `plot_pod_modes()` | Spatial mode shapes |
| ROM Validation | 5 | `_plot_comparison()` | Truth vs ROM comparison |
| Error vs Dimension | 5 | `plot_error_vs_dimension()` | Paper reproduction |
| Noise Analysis | Ext | `plot_noise_analysis_results()` | Regularization robustness |
| Regularization | Ext | `plot_regularization_comparison()` | Hyperparameter tuning |

### **Available but Not Auto-Called:**

| Visual | Module | Function | Description |
|--------|--------|----------|-------------|
| Phase Space | `visualization.py` | `plot_phase_space()` | s₁ vs s₂ trajectories |
| Parameter Study | `visualization.py` | `plot_parameter_study_results()` | Parameter sweep analysis |
| Interactive Plots | `visualization.py` | `create_interactive_solution_plot()` | Plotly dashboard |
| Animations | `visualization.py` | `create_animation()` | GIF animations |

---

## 🎮 **Manual Visual Generation**

If you want to generate specific visuals outside the main pipeline:

### **Example: Create Phase Space Plot**
```python
# After running main.py to get data
from src.visualization import FitzHughNagumoVisualizer
from src.fitzhugh_nagumo_solver import FitzHughNagumoSolver

# Create solver and get solution
solver = FitzHughNagumoSolver(nx=50)
solution = solver.solve(...)

# Create visualizer and plot
viz = FitzHughNagumoVisualizer(solver.x)
viz.plot_phase_space(solution)
```

### **Example: Create Animation**
```python
# Create animation
viz.create_animation(solution, "my_animation.gif", fps=15)
```

### **Example: Interactive Dashboard**
```python
# Create Plotly interactive plot
viz.create_interactive_solution_plot(solution)
```

---

## 🔧 **Troubleshooting Visual Issues**

### **No Plots Appearing**

**Check Backend:**
```python
import matplotlib
print(matplotlib.get_backend())
```

**Set Backend (if needed):**
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### **Plots Appear but Close Immediately**

**Add to end of script:**
```python
import matplotlib.pyplot as plt
plt.show(block=True)  # Keep plots open
```

### **Save Plots Instead of Display**

**Modify visualization functions:**
```python
# In any plotting function, replace:
plt.show()

# With:
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## 📁 **File Locations Summary**

```
FitzHugh-Nagumo System/
├── main.py                 # Controls when visuals appear
├── src/
│   ├── visualization.py    # Main visualization functions
│   ├── pod_reduction.py    # POD analysis plots
│   ├── validation.py       # ROM validation plots
│   └── noise_extension.py  # Noise analysis plots
├── VISUALIZATION_GUIDE.md  # Detailed plot interpretation
└── VISUAL_LOCATIONS.md     # This file (where to find plots)
```

---

## 🎯 **Quick Test**

To see a visual right now:

```bash
cd "FitzHugh-Nagumo System"
source venv/bin/activate
python main.py --phase 1
```

This will run for ~5 seconds, then a popup window will appear showing the sample training data visualization. Close the window to see the program complete.

---

**The key point**: Visuals are **automatic popup windows** that appear during execution phases. You don't need to look for saved files - they appear as interactive windows that you can explore, save, or close.