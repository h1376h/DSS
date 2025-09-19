#!/usr/bin/env python3
"""
Healthcare DSS Architecture Figures Generator Version
===============================================================

This script generates improved, professional figures for the Healthcare_DSS_Architecture.md document
with better spacing, readability, and visual hierarchy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Ellipse
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_figure_1_1():
    """Figure 1.1: The Decision-Making/Modeling Process for Healthcare DSS"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Define phases with better spacing to prevent overlap
    phases = {
        'Intelligence': {
            'pos': (0.2, 0.8),
            'size': (0.15, 0.1),
            'color': '#E74C3C',
            'description': 'Problem\nIdentification'
        },
        'Design': {
            'pos': (0.8, 0.8),
            'size': (0.15, 0.1),
            'color': '#3498DB',
            'description': 'Model\nConstruction'
        },
        'Choice': {
            'pos': (0.8, 0.2),
            'size': (0.15, 0.1),
            'color': '#2ECC71',
            'description': 'Solution\nSelection'
        },
        'Implementation': {
            'pos': (0.2, 0.2),
            'size': (0.15, 0.1),
            'color': '#F39C12',
            'description': 'Solution\nDeployment'
        }
    }
    
    # Draw phases with cleaner styling
    for phase_name, phase_info in phases.items():
        x, y = phase_info['pos']
        w, h = phase_info['size']
        
        # Create clean rounded rectangle
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=phase_info['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add phase title and description in one clean text
        ax.text(x, y, f"{phase_name}\n{phase_info['description']}", 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Draw main flow arrows with improved positioning
    arrow_positions = [
        ((0.275, 0.8), (0.725, 0.8)),  # Intelligence to Design
        ((0.8, 0.75), (0.8, 0.25)),  # Design to Choice
        ((0.725, 0.2), (0.275, 0.2)),  # Choice to Implementation
        ((0.2, 0.25), (0.2, 0.75)),  # Implementation to Intelligence
    ]
    
    for start, end in arrow_positions:
        arrow = patches.FancyArrowPatch(
            start, end,
            arrowstyle='->', mutation_scale=15,
            color='black', linewidth=2.5,
            zorder=4
        )
        ax.add_patch(arrow)
    
    # Add simplified title
    ax.set_title('Figure 1.1: Decision-Making Process for Healthcare DSS', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_figures/Figure_1_1_Decision_Making_Process.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Figure 1.1: Decision-Making Process for Healthcare DSS")

def create_figure_1_2():
    """Figure 1.2: Decision Support Frameworks for Healthcare"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Create simplified 3x3 matrix with shorter text
    matrix_data = [
        ['Patient Billing\nStandard Protocols', 
         'Resource Allocation\nStaff Scheduling', 
         'Strategic Planning\nCapital Investment'],
        ['Treatment Planning\nCare Coordination', 
         'Budget Planning\nQuality Improvement', 
         'Healthcare Policy\nTechnology Adoption'],
        ['Novel Treatment\nResearch Decisions', 
         'Crisis Management\nChange Management', 
         'Vision Setting\nRegulatory Strategy']
    ]
    
    # Colors for each cell with better contrast
    colors = [
        ['#FFE5E5', '#FFCCCC', '#FFB3B3'],  # Light reds
        ['#E5F3FF', '#CCE7FF', '#B3DBFF'],  # Light blues
        ['#E5FFE5', '#CCFFCC', '#B3FFB3']   # Light greens
    ]
    
    # Labels
    row_labels = ['Structured', 'Semistructured', 'Unstructured']
    col_labels = ['Operational\nControl', 'Managerial\nControl', 'Strategic\nPlanning']
    
    # Create matrix with better spacing to prevent cropping - adjust layout
    cell_width = 0.20
    cell_height = 0.20
    start_x = 0.30
    start_y = 0.30
    
    for i in range(3):
        for j in range(3):
            x = start_x + j * (cell_width + 0.05)
            y = start_y + i * (cell_height + 0.05)
            
            # Cell background with better border
            rect = Rectangle((x, y), cell_width, cell_height, 
                           facecolor=colors[i][j], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Cell content with better text positioning
            ax.text(x + cell_width/2, y + cell_height/2, matrix_data[i][j], 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Add row labels closer to the matrix
    for i, label in enumerate(row_labels):
        y = start_y + i * (cell_height + 0.05) + cell_height/2
        ax.text(start_x - 0.08, y, label, ha='center', va='center', fontsize=11, fontweight='bold',
                rotation=90, bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
    
    # Add column labels closer to the matrix
    for j, label in enumerate(col_labels):
        x = start_x + j * (cell_width + 0.05) + cell_width/2
        ax.text(x, start_y - 0.08, label, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
    
    # Add axis labels with proper centering
    # "Types of Control" centered below the grey boxes
    matrix_center_x = start_x + 1.5 * (cell_width + 0.05)  # Center of the 3x3 matrix
    ax.text(matrix_center_x, start_y - 0.15, 'Types of Control', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # "Degree of Structuredness" centered to the left
    matrix_center_y = start_y + 1.5 * (cell_height + 0.05)  # Center of the 3x3 matrix
    ax.text(start_x - 0.15, matrix_center_y, 'Degree of\nStructuredness', ha='center', va='center', 
            fontsize=12, fontweight='bold', rotation=90)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Figure 1.2: Decision Support Frameworks for Healthcare', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('generated_figures/Figure_1_2_Decision_Support_Frameworks.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Figure 1.2: Decision Support Frameworks for Healthcare")

def create_figure_1_3():
    """Figure 1.3: Key Characteristics and Capabilities of Healthcare DSS"""
    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    
    # Central DSS hub with better sizing
    center_circle = Circle((0.5, 0.5), 0.07, facecolor='#E74C3C', edgecolor='black', linewidth=3)
    ax.add_patch(center_circle)
    ax.text(0.5, 0.5, 'Healthcare\nDSS Hub', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Simplified capabilities with shorter text
    capabilities = [
        'Semistructured\nProblems',
        'All Managerial\nLevels',
        'Individual &\nGroup Support',
        'Sequential\nDecisions',
        'All Decision\nPhases',
        'Variety of\nProcesses',
        'Adaptability\n& Flexibility',
        'User-Friendly\nInterface',
        'Improved\nEffectiveness',
        'Decision Maker\nControl',
        'Easy Development\nby Users',
        'Modeling &\nAnalysis',
        'Multiple Data\nSources',
        'Standalone &\nWeb-based'
    ]
    
    # Calculate positions for 14 spokes with better spacing
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    radius = 0.22
    
    for i, (angle, capability) in enumerate(zip(angles, capabilities)):
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        
        # Draw spoke with smaller arrows
        ax.plot([0.5, x], [0.5, y], 'k-', linewidth=1.5, alpha=0.6)
        
        # Add capability box with improved styling
        rect = FancyBboxPatch(
            (x - 0.07, y - 0.05), 0.14, 0.10,
            boxstyle="round,pad=0.01",
            facecolor='lightblue',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.95
        )
        ax.add_patch(rect)
        
        # Add capability text with better positioning
        ax.text(x, y, capability, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add capability number with better positioning
        num_x = x + 0.08 * np.cos(angle)
        num_y = y + 0.08 * np.sin(angle)
        num_circle = Circle((num_x, num_y), 0.02, facecolor='red', edgecolor='black', linewidth=1)
        ax.add_patch(num_circle)
        ax.text(num_x, num_y, str(i+1), ha='center', va='center', 
                fontsize=7, fontweight='bold', color='white')
    
    # Add simplified legend
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='DSS Hub'),
        mpatches.Patch(color='lightblue', label='Capabilities')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Figure 1.3: Key Characteristics and Capabilities of Healthcare DSS', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('generated_figures/Figure_1_3_DSS_Capabilities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Figure 1.3: Key Characteristics and Capabilities of Healthcare DSS")

def create_figure_1_4():
    """Figure 1.4: Schematic View of Healthcare DSS"""
    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    
    # Define subsystems with improved positioning
    subsystems = {
        'Data Management': {
            'pos': (0.2, 0.8),
            'size': (0.22, 0.12),
            'color': '#E74C3C',
            'components': ['EHR Integration', 'Data Warehouse', 'ETL Processes']
        },
        'Model Management': {
            'pos': (0.8, 0.8),
            'size': (0.22, 0.12),
            'color': '#3498DB',
            'components': ['Predictive Models', 'ML Algorithms', 'Model Training']
        },
        'Knowledge Management': {
            'pos': (0.2, 0.2),
            'size': (0.22, 0.12),
            'color': '#2ECC71',
            'components': ['Clinical Guidelines', 'Expert Systems', 'Rule Engine']
        },
        'User Interface': {
            'pos': (0.8, 0.2),
            'size': (0.22, 0.12),
            'color': '#F39C12',
            'components': ['Dashboards', 'Reports', 'Mobile Apps']
        }
    }
    
    # Draw subsystems with improved styling
    for sub_name, sub_info in subsystems.items():
        x, y = sub_info['pos']
        w, h = sub_info['size']
        
        # Main subsystem box with better styling
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=sub_info['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Subsystem title at the top
        ax.text(x, y + 0.04, sub_name, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        # Components below the title
        for i, component in enumerate(sub_info['components']):
            comp_y = y - 0.02 + i * 0.025
            ax.text(x, comp_y, f"• {component}", 
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Central integration hub with better positioning
    center_circle = Circle((0.5, 0.5), 0.05, facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(center_circle)
    ax.text(0.5, 0.5, 'Integration\nHub', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connections with improved clarity
    connections = [
        ((0.31, 0.8), (0.45, 0.5)),  # Data Management to Hub
        ((0.69, 0.8), (0.55, 0.5)),  # Model Management to Hub
        ((0.31, 0.2), (0.45, 0.5)),  # Knowledge Management to Hub
        ((0.69, 0.2), (0.55, 0.5)),  # User Interface to Hub
    ]
    
    for start, end in connections:
        arrow = patches.FancyArrowPatch(
            start, end,
            arrowstyle='->', mutation_scale=12,
            color='black', linewidth=2,
            zorder=3
        )
        ax.add_patch(arrow)
    
    # Add external data sources with better positioning
    external_sources = ['EHR Systems', 'Lab Systems', 'Imaging Systems']
    for i, source in enumerate(external_sources):
        x = 0.2 + i * 0.3
        y = 0.95
        rect = Rectangle((x - 0.05, y - 0.025), 0.1, 0.05, 
                        facecolor='lightgray', edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, source, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Arrow to Data Management with better positioning
        arrow = patches.FancyArrowPatch(
            (x, y - 0.025), (0.2, 0.86),
            arrowstyle='->', mutation_scale=10,
            color='blue', linewidth=1.5
        )
        ax.add_patch(arrow)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Figure 1.4: Schematic View of Healthcare DSS', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('generated_figures/Figure_1_4_DSS_Architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Figure 1.4: Schematic View of Healthcare DSS")

def create_figure_2_1():
    """Figure 2.1: The AI Tree for Healthcare Applications"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Draw tree trunk with better styling
    trunk = Rectangle((0.45, 0.3), 0.1, 0.4, facecolor='#8B4513', edgecolor='black', linewidth=2)
    ax.add_patch(trunk)
    ax.text(0.5, 0.5, 'AI\nTechnologies', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Draw roots (foundations) with better positioning
    foundations = {
        'Biology &\nNeurology': (0.2, 0.15, '#8B4513'),
        'Psychology &\nCognition': (0.5, 0.15, '#8B4513'),
        'Linguistics': (0.8, 0.15, '#8B4513')
    }
    
    for foundation, (x, y, color) in foundations.items():
        root = Rectangle((x - 0.08, y - 0.05), 0.16, 0.1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(root)
        ax.text(x, y, foundation, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Draw root lines
        ax.plot([x, 0.5], [y + 0.05, 0.3], 'k-', linewidth=4)
    
    # Draw main branches (AI technologies) with better spacing
    branches = {
        'Computer\nVision': (0.15, 0.6, '#E74C3C'),
        'Machine\nLearning': (0.35, 0.75, '#3498DB'),
        'Expert\nSystems': (0.65, 0.75, '#2ECC71'),
        'Natural Language\nProcessing': (0.85, 0.6, '#F39C12')
    }
    
    for branch, (x, y, color) in branches.items():
        # Branch line
        ax.plot([0.5, x], [0.7, y], 'k-', linewidth=5)
        
        # Branch circle
        circle = Circle((x, y), 0.06, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, branch, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Draw leaves (applications) with better organization
    applications = {
        'Medical Image\nAnalysis': (0.1, 0.8, '#2ECC71'),
        'Predictive\nModels': (0.25, 0.9, '#E74C3C'),
        'Diagnostic\nAssistants': (0.75, 0.9, '#F39C12'),
        'Health\nChatbots': (0.9, 0.8, '#9B59B6'),
        'Pattern\nRecognition': (0.05, 0.7, '#1ABC9C'),
        'Treatment\nOptimization': (0.4, 0.85, '#E67E22'),
        'Clinical Notes\nAnalysis': (0.95, 0.7, '#34495E')
    }
    
    for app, (x, y, color) in applications.items():
        # Leaf with better styling
        leaf = Ellipse((x, y), 0.08, 0.06, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(leaf)
        ax.text(x, y, app, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Connect to branch
        if x < 0.4:
            ax.plot([x, 0.15], [y, 0.6], 'k-', linewidth=2)
        elif x < 0.6:
            ax.plot([x, 0.35], [y, 0.75], 'k-', linewidth=2)
        else:
            ax.plot([x, 0.85], [y, 0.6], 'k-', linewidth=2)
    
    # Add simplified legend
    legend_elements = [
        mpatches.Patch(color='#8B4513', label='Foundational Sciences'),
        mpatches.Patch(color='#E74C3C', label='AI Technologies'),
        mpatches.Patch(color='#2ECC71', label='Healthcare Applications')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Figure 2.1: The AI Tree for Healthcare Applications', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('generated_figures/Figure_2_1_AI_Tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Figure 2.1: The AI Tree for Healthcare Applications")

def create_figure_2_2():
    """Figure 2.2: Cost of Human Work vs. AI Work in Healthcare"""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    
    # Time axis
    time = np.linspace(0, 10, 100)
    
    # Human cost (rising) with more realistic curve
    human_cost = 50 + 3 * time + 0.15 * time**2
    
    # AI cost (falling) with more realistic curve
    ai_cost = 300 * np.exp(-0.25 * time) + 15
    
    # Plot lines with improved styling
    ax.plot(time, human_cost, 'r-', linewidth=3, label='Human Work Cost', marker='o', markersize=3, markevery=20)
    ax.plot(time, ai_cost, 'b-', linewidth=3, label='AI/Robotics Cost', marker='s', markersize=3, markevery=20)
    
    # Find crossover point
    crossover_idx = np.where(np.abs(human_cost - ai_cost) == np.min(np.abs(human_cost - ai_cost)))[0][0]
    crossover_time = time[crossover_idx]
    crossover_cost = human_cost[crossover_idx]
    
    # Mark crossover point with improved styling
    ax.plot(crossover_time, crossover_cost, 'go', markersize=10, label=f'Crossover Point\n({crossover_time:.1f} years)', markeredgecolor='black', markeredgewidth=2)
    ax.axvline(x=crossover_time, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=crossover_cost, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add improved annotations
    ax.annotate(f'AI becomes\ncost-effective\nat {crossover_time:.1f} years', 
                xy=(crossover_time, crossover_cost), 
                xytext=(crossover_time + 1.5, crossover_cost + 15),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Time (Years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cost per Task ($)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 2.2: Cost of Human Work vs. AI Work in Healthcare', 
                 fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=1)
    
    # Set better axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(human_cost) * 1.1)
    
    plt.tight_layout()
    plt.savefig('generated_figures/Figure_2_2_Cost_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created Figure 2.2: Cost of Human Work vs. AI Work in Healthcare")

def main():
    """Generate figures"""
    import os
    
    # Create generated_figures directory if it doesn't exist
    if not os.path.exists('generated_figures'):
        os.makedirs('generated_figures')
    
    print("Generating Healthcare DSS Architecture Figures...")
    print("=" * 60)
    
    # Generate Chapter 1 figures
    create_figure_1_1()
    create_figure_1_2()
    create_figure_1_3()
    create_figure_1_4()
    
    # Generate Chapter 2 figures
    create_figure_2_1()
    create_figure_2_2()
    
    print("=" * 60)
    print("figure generation completed!")
    print("All figures saved in the 'generated_figures' directory.")

if __name__ == "__main__":
    main()
