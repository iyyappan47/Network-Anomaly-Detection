---
name: Cyber Defense Sentinel
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#414755'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#727786'
  outline-variant: '#c1c6d7'
  surface-tint: '#0059c7'
  primary: '#0057c2'
  on-primary: '#ffffff'
  primary-container: '#006ef2'
  on-primary-container: '#fefcff'
  inverse-primary: '#afc6ff'
  secondary: '#4c607a'
  on-secondary: '#ffffff'
  secondary-container: '#cadefd'
  on-secondary-container: '#4e627c'
  tertiary: '#006947'
  on-tertiary: '#ffffff'
  tertiary-container: '#00855b'
  on-tertiary-container: '#f5fff6'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d9e2ff'
  primary-fixed-dim: '#afc6ff'
  on-primary-fixed: '#001a43'
  on-primary-fixed-variant: '#004398'
  secondary-fixed: '#d2e4ff'
  secondary-fixed-dim: '#b3c8e6'
  on-secondary-fixed: '#051c33'
  on-secondary-fixed-variant: '#344861'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 2rem
    fontWeight: '700'
    lineHeight: 2.5rem
  headline-lg:
    fontFamily: Inter
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: 2rem
  headline-md:
    fontFamily: Inter
    fontSize: 1.25rem
    fontWeight: '600'
    lineHeight: 1.75rem
  headline-sm:
    fontFamily: Inter
    fontSize: 1.125rem
    fontWeight: '600'
    lineHeight: 1.5rem
  body-lg:
    fontFamily: Inter
    fontSize: 1rem
    fontWeight: '400'
    lineHeight: 1.5rem
  body-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: 1.25rem
  body-sm:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: '400'
    lineHeight: 1rem
  label-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '500'
    lineHeight: 1.25rem
  label-sm:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: '500'
    lineHeight: 1rem
  mono-data:
    fontFamily: Inter
    fontSize: 0.8125rem
    fontWeight: '400'
    lineHeight: 1.25rem
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.25rem
  margin: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style

The design system is constructed for mission-critical enterprise cybersecurity monitoring, AI-driven network anomaly detection, and operational threat hunting. Its primary goal is to establish uncompromising trust, immediate analytical clarity, and cognitive comfort during prolonged investigative sessions.

### Aesthetic Persona
- **Corporate & Utilitarian Modernism:** Drawing from balanced enterprise systems like Ant Design and refined Tailwind dashboard architectures. It avoids superficial ornament in favor of structural clarity, precise geometric bounding, and deterministic visual hierarchy.
- **Deep Slate Anchoring vs. Open Canvas:** High-contrast persistent deep navy sidebar framing contrasted against an airy, low-fatigue slate-blue workspace.
- **High-Fidelity Telemetry:** Data-dense metrics, interactive multi-class confusion matrices, precision classification tables, and categorical telemetry badges designed to surface anomalies instantly without sensory overwhelm.

### User Emotional Response
- **In Control:** Every layout decision prioritizes high scanability, crisp tabular alignment, and zero ambiguous state transitions.
- **Vigilant yet Calm:** Critical alerts (coral/red) pierce through an otherwise stable, trustworthy neutral landscape of calm blues and neutral slates.

## Colors

The palette leverages a structured semantic tier built for high-contrast data visualization, clear operational states, and rapid visual triage.

### Dominant Foundations
- **Canvas / Background (`#F4F7FB`):** A soft, low-luminance blue-slate that prevents ocular fatigue while keeping white card containers distinctly framed.
- **Sidebar Background (`#0D233A` to `#0B192C`):** A commanding deep oceanic navy providing structural anchoring and high-contrast navigation.
- **Surface Containers (`#FFFFFF`):** Pure white modular card surfaces, equipped with `#E2E8F0` structural outlines.

### Brand & Interactive Accents
- **Primary Brand (`#1677FF`):** Vibrant cobalt blue used for active sidebar indicators, primary operational buttons, process steps, and focused telemetry cards.
- **Hover/Active Primary (`#0958D9`):** Deep cobalt for active press states and high-emphasis boundaries.

### Semantic Status & Telemetry
- **Success / Normal Traffic (`#10B981` / `#52C41A`):** Vivid emerald green used for normal packet volumes, high precision/recall marks, and successful validation checks. Paired with soft tint backgrounds (`#ECFDF5`).
- **Critical / Threat / Attack (`#EF4444` / `#FF4D4F`):** Vivid coral red used for intrusion alerts, attack records, and critical matrix cells. Paired with soft tint backgrounds (`#FEF2F2`).
- **Informational / Machine Learning (`#8B5CF6` / `#722ED1`):** Precision purple used for model statuses, F1-scores, and random forest pipeline steps.
- **Warning / Protocol Multi-Class (`#F59E0B`):** Warm amber used for intermediate traffic classes and probe warnings.

### Functional Neutrals
- **Heading & Primary Text (`#0F172A` / `#1E293B`):** Deep slate-black for effortless tabular and metric legibility.
- **Muted & Subtitle Text (`#64748B`):** Balanced cool slate for metadata, axis marks, timestamps, and secondary card descriptions.
- **Card Borders & Dividers (`#E2E8F0` / `#F1F5F9`):** Fine architectural line dividers ensuring structural containment.

## Typography

Typography relies entirely on the neutral, neo-grotesque structural clarity of Inter. It delivers high legibility across tabular datasets, IP routing telemetry, and rapid-glance dashboard KPIs.

### Typographic Hierarchy & Application
- **KPI Metrics (`headline-xl` / 32px / Bold):** Used for single key figures (e.g., `82,332`, `0.952`). High contrast, uncluttered, and paired immediately with an upper or lower descriptive subtitle.
- **View Headers (`headline-lg` / 24px / Semi-bold):** Used for core page headings (`Overview`, `Model Performance`, `Detection Results`).
- **Section Titles (`headline-sm` / 18px / Semi-bold):** Identifies widget boundaries such as `Confusion Matrix`, `Traffic Distribution`, and `Record Details`.
- **Tabular & Telemetry Body (`body-md` / 14px):** Primary reading level for tables, form selectors, and step labels.
- **Status & Micro Badges (`label-sm` / 12px / Medium):** Pill badges, confidence percentages, and metadata timestamps. Tabular numerals (`tnum`) must be enforced across all numeric columns and metric displays to avoid horizontal jitter during live telemetry updates.

## Layout & Spacing

The layout is built upon an asymmetric enterprise split: a fixed-width left navigation rail combined with a fluid, multi-column dashboard body.

### Canvas Grid Model
- **Navigation Sidebar:** Fixed width of `240px` (collapsible to `64px` on tablet), anchored to the left viewport boundary.
- **Top Utility Header:** Persistent bar spanning the remaining canvas width, with an internal vertical height of `56px` to house pathing, live timestamp telemetry, and profile access.
- **Dashboard Grid:** 12-column responsive fluid grid operating with `1.25rem` (20px) gutters and a global canvas margin of `1.5rem` (24px).
  - **KPI Row:** Typically 4 columns (`col-span-3` each).
  - **Analytical Views:** Balanced splits (`col-span-6` / `col-span-6` or `col-span-8` / `col-span-4` for inspection panels).

### Responsive Adaptation
- **Desktop (>= 1280px):** Full expanded sidebar, 4-tier metric grid, side-by-side data table and record drawer.
- **Tablet (768px - 1279px):** Icon-collapsed sidebar, 2x2 metric card grid, full-width data tables with horizontal scroll containment.
- **Mobile (< 768px):** Off-canvas drawer navigation, single-column KPI stacking, and prioritized card views for anomalous rows.

## Elevation & Depth

Depth is established primarily through clean surface stratification, crisp low-contrast perimeter strokes, and subtle ambient shadows rather than dramatic Z-axis stacking.

### Surface Elevation Levels
- **Layer 0 (Canvas Base):** Background wash at `#F4F7FB`. Completely flat with zero elevation.
- **Layer 1 (Cards & Data Containers):** Pure `#FFFFFF` surfaces bounded by a crisp `1px solid #E2E8F0` stroke. Paired with a soft ambient shadow: `0px 1px 3px rgba(15, 23, 42, 0.04), 0px 4px 6px -1px rgba(15, 23, 42, 0.02)`.
- **Layer 2 (Dropdowns, Floating Overlays, Modals):** White containers bordered with `#CBD5E1` casting an ambient shadow: `0px 10px 15px -3px rgba(15, 23, 42, 0.08), 0px 4px 6px -2px rgba(15, 23, 42, 0.04)`.
- **Layer Active (Navigation Pill):** High-contrast solid primary blue (`#1677FF`) resting flush within the deep navy sidebar, creating deliberate figure-ground dominance.

## Shapes

The design system adopts a balanced roundedness configuration (`level 2`), harmonizing modern approachable software design with enterprise engineering discipline.

### Geometry Hierarchy
- **Standard UI Elements & Inputs (`0.5rem` / 8px):** Standard buttons, text search fields, dropdown selects, and modular metric card shells.
- **Surface Cards & Data Grids (`rounded-lg` / `1rem` / 16px):** Primary analytics cards, chart bounding boxes, and wizard step sections.
- **Status Badges & Pill Indicators (`rounded-full` / 9999px):** Anomaly alerts (`Normal`, `Attack`), protocol pills (`TCP`, `UDP`), confidence tags, and the active navigation selection indicator.

## Components

### Buttons
- **Primary:** Solid `#1677FF` background, `#FFFFFF` text, `0.5rem` border radius, padding `0.5rem 1rem`. Subtly transitions to `#0958D9` on hover.
- **Secondary / Ghost:** White background, `#E2E8F0` border, `#0F172A` text. Hover shifts to `#F8FAFC`.
- **Action Links:** Minimal text button in `#1677FF` with semi-bold weighting for inline actions such as table `View` triggers.

### Metric KPI Cards
- Standardized card container with an internal horizontal layout:
  - Left/Top-left: 40x40px rounded icon container with soft pastel tint matching the metric's semantic theme (e.g., `#EFF6FF` for blue total records, `#ECFDF5` for green normal records, `#FEF2F2` for red attacks, `#F5F3FF` for purple model status).
  - Primary metric numeral set in `headline-xl` (`#0F172A`), followed by a subtitle or percentage delta tag.

### Data Tables
- **Header:** Background `#F8FAFC`, uppercase label style (`label-sm`), `#64748B` typography, bottom border `1px solid #E2E8F0`.
- **Rows:** Alternating hover highlight (`#F8FAFC`), standard height `48px`, border-bottom `1px solid #F1F5F9`.
- **Status Cells:** Pill badges containing a leading status dot or pure solid badge (`#ECFDF5` background with `#047857` text for Normal; `#FEF2F2` background with `#B91C1C` text for Attack).

### Confusion Matrix (2x2 Grid)
- Custom analytical component with high-contrast color coding:
  - True Negatives / True Positives: Soft green fill (`#DCFCE7`) with bold green text.
  - False Positives / False Negatives: Soft coral red fill (`#FEE2E2`) with bold red text.
  - Bounded by clean internal `#FFFFFF` grid separators.

### Workflow & Wizard Steppers
- Horizontal circular node progression (`1. Upload`, `2. Validate`, `3. Preprocess`, `4. Predict`, etc.):
  - Active/Complete: Solid `#1677FF` node with white numeric index or checkmark, linked by `#1677FF` solid connector lines.
  - Incomplete: White node with `#CBD5E1` border and muted text, connected by dashed `#E2E8F0` tracks.

### File Upload Zone
- Large dashed bounding box (`2px dashed #93C5FD`), soft blue-tinted background (`#F0F7FF`), centered cloud illustration/icon, supported formats notice, and a prominent centered primary upload action.