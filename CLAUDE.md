# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a GitHub Pages portfolio website for Shafkat Rahman, an AI consultant and data scientist. The site showcases multiple research and engineering projects across machine learning, computational physics, and quantitative finance.

## Project Structure

The repository follows a simple static website structure:

- **Root HTML files**: `index.html` (main portfolio), `about.html`, `simple-test.html`
- **Project subdirectories**: Each major project has its own directory containing:
  - `index.html` - Project showcase page
  - `assets/` or `visualizations/` - Images, plots, and visual content
  - Source code files (Python scripts, data files)
  - Documentation files (README.md, technical guides)

## Key Project Categories

### Featured Projects (in order of prominence on site):
1. **gPINN** - Gradient-enhanced Physics-Informed Neural Networks for geothermal exploration
2. **Drag Coefficient Prediction** - Physics-guided neural networks for CFD applications
3. **FitzHugh-Nagumo Lift & Learn** - Reduced-order modeling with operator inference
4. **Gardner Capacity Puzzle** - Statistical physics investigation of perceptron storage
5. **Topological Photonic Crystal Optimizer** - Multi-objective ML for computational photonics
6. **GRPO Healthcare AI** - Group robust policy optimization for medical resource allocation
7. **Chaotic Systems Analysis** - Mathematical toolkit for nonlinear dynamical systems
8. **Financial projects** - Multiple quantitative finance and market analysis tools

## Development Workflow

### Static Site Deployment
- Site deploys automatically via GitHub Pages from the main branch
- No build process required - direct HTML/CSS/JavaScript files
- Changes to HTML files are immediately reflected on the live site

### Content Updates
- Project showcases are self-contained in their respective directories
- Main portfolio updates require editing `index.html`
- Visual assets are organized in project-specific `assets/` or `visualizations/` directories

### Project Documentation Structure
- Most projects include comprehensive README files and technical documentation
- Visualization scripts (like `generate_visualizations.py`) create project showcase materials
- Projects maintain their own LICENSE and requirements files where applicable

## Technical Stack

- **Frontend**: Pure HTML, CSS, JavaScript (no frameworks)
- **Styling**: Custom CSS with monospace font aesthetic
- **Responsive Design**: Media queries for mobile compatibility
- **Navigation**: Smooth scrolling JavaScript for single-page experience

## Content Management

### Adding New Projects
1. Create new directory with project name
2. Include `index.html` for project showcase
3. Add `assets/` directory for visual content
4. Update main `index.html` to include project in featured list
5. Maintain consistent styling and structure

### Project Showcase Requirements
- Professional technical descriptions
- Performance metrics and achievements
- Links to live demos and source code
- High-quality visualizations and plots

## Common Tasks

Since this is a static site with no build process:
- **Local Development**: Open HTML files directly in browser
- **Content Updates**: Edit HTML files directly
- **Asset Management**: Add images/plots to appropriate project directories
- **Testing**: Use `simple-test.html` for quick verification

## Site Architecture Notes

- Main portfolio uses sections with anchor navigation
- Each project maintains independent styling while following site aesthetics
- Responsive design prioritizes mobile-first approach
- SEO optimization included with meta tags and structured content