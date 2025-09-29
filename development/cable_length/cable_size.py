import streamlit as st

class CableSizingCalculator:
    def __init__(self):
        self.cable_data = {
            1: {'current': 16, 'volt_drop': 116.9, 'earth_size': 1},
            1.5: {'current': 20, 'volt_drop': 74.9, 'earth_size': 1.5},
            2.5: {'current': 28, 'volt_drop': 40.9, 'earth_size': 2.5},
            4: {'current': 38, 'volt_drop': 25.5, 'earth_size': 2.5},
            6: {'current': 48, 'volt_drop': 17, 'earth_size': 2.5},
            10: {'current': 66, 'volt_drop': 10.1, 'earth_size': 4},
            16: {'current': 88, 'volt_drop': 6.4, 'earth_size': 6},
            25: {'current': 119, 'volt_drop': 4, 'earth_size': 6},
            35: {'current': 147, 'volt_drop': 2.9, 'earth_size': 10},
            50: {'current': 180, 'volt_drop': 2.2, 'earth_size': 16},
            70: {'current': 229, 'volt_drop': 1.5, 'earth_size': 25},
            95: {'current': 283, 'volt_drop': 1.1, 'earth_size': 25},
            120: {'current': 330, 'volt_drop': 0.9, 'earth_size': 35},
            150: {'current': 377, 'volt_drop': 0.7, 'earth_size': 50},
            185: {'current': 436, 'volt_drop': 0.6, 'earth_size': 70},
            240: {'current': 517, 'volt_drop': 0.5, 'earth_size': 95},
            300: {'current': 594, 'volt_drop': 0.4, 'earth_size': 120},
            400: {'current': 685, 'volt_drop': 0.4, 'earth_size': 120},
            500: {'current': 779, 'volt_drop': 0.4, 'earth_size': 120}
        }

    def calculate_cable(self, load_current, distance, voltage, max_voltage_drop, phase):
        """Calculate suitable cable size based on given parameters"""
        distance_km = distance / 1000

        for size, data in self.cable_data.items():
            # Current rating check
            current_rating = data['current']
            current_ok = current_rating >= load_current

            # Voltage drop calculation
            if phase == "3 Phase AC":
                voltage_drop = load_current * distance_km * data['volt_drop']
            else:  # Single Phase AC or DC
                voltage_drop = load_current * distance_km * (data['volt_drop'] * 2)

            voltage_drop_percent = (voltage_drop / voltage) * 100
            voltage_ok = voltage_drop_percent <= max_voltage_drop

            if current_ok and voltage_ok:
                return {
                    'size': size,
                    'current_rating': current_rating,
                    'earth_size': data['earth_size'],
                    'voltage_drop_percent': voltage_drop_percent,
                    'voltage_drop': voltage_drop
                }
        return None
    
    def get_neutral_core(self, phase, cable_size):
        """Get neutral core value based on phase type"""
        if phase == "3 Phase AC":
            return "No neutral"
        else:  # Single Phase AC or DC
            return f"{cable_size} mm²"
    
    def get_calculation_results(self, load_current, distance, voltage, max_voltage_drop, phase, conductor):
        """Get formatted calculation results"""
        suitable_cable = self.calculate_cable(load_current, distance, voltage, max_voltage_drop, phase)
        
        if suitable_cable:
            neutral_core = self.get_neutral_core(phase, suitable_cable['size'])
            
            # Determine status indicators
            current_status = "✅ Safe" if suitable_cable['current_rating'] >= load_current else "❌ Unsafe"
            voltage_status = "✅ Safe" if suitable_cable['voltage_drop_percent'] <= max_voltage_drop else "❌ Exceeds Limit"
            
            return {
                'success': True,
                'suggested_cable_size': f"{suitable_cable['size']} mm²",
                'neutral_core': neutral_core,
                'earth_size': f"{suitable_cable['earth_size']} mm²",
                'current_rating': f"{suitable_cable['current_rating']} A (Load: {load_current} A) - {current_status}",
                'voltage_drop': f"{suitable_cable['voltage_drop_percent']:.1f}% (Limit: {max_voltage_drop}%) - {voltage_status}",
                'conductor_material': conductor
            }
        else:
            return {'success': False}

# Streamlit App
def main():
    # Page configuration
    st.set_page_config(page_title="Cable Sizing Calculator", page_icon="⚡", layout="wide")
    
    # Initialize calculator
    calculator = CableSizingCalculator()
    
    # Title
    st.title("⚡ Cable Sizing Calculator")
    st.markdown("*Based on Australian/New Zealand Standard (AS/NZS 3008)*")
    
    # Parameters Section
    st.header("⚙️ Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        phase = st.selectbox("Phase Type", ["3 Phase AC", "Single Phase AC", "DC"])
        voltage = st.number_input("Voltage (V)", value=400 if phase == "3 Phase AC" else 230)
        load_current = st.number_input("Load Current (A)", min_value=1, max_value=1000, value=250)
    
    with col2:
        distance = st.number_input("Cable Length (m)", min_value=1, max_value=500, value=40)
        max_voltage_drop = st.slider("Max Voltage Drop (%)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        conductor = st.selectbox("Conductor Material", ["Copper", "Aluminium"])
    
    with col3:
        insulation = st.selectbox("Insulation Type", ["XLPE X-90 Standard 90°", "PVC V-90 Standard 75°"])
        cable_type = st.selectbox("Cable Configuration", ["Multi-core 3C+E", "Multi-core 4C+E", "Single-cores 3x1C+E"])
        installation_method = st.selectbox("Installation Method", ["Direct buried", "In conduit", "On cable tray", "Clipped direct", "Free air"])
    
    # Calculate button
    st.divider()
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col2:
        calculate_btn = st.button("🔍 Calculate Cable Size", type="primary", use_container_width=True)
    
    # Results Section
    st.header("📊 Calculation Results")
    
    if calculate_btn:
        results = calculator.get_calculation_results(load_current, distance, voltage, max_voltage_drop, phase, conductor)
        
        if results['success']:
            st.success("**Suitable Cable Found!**")
            st.write("---")
            
            # Display results in columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.markdown("**Suggested Cable Size**")
                st.info(results['suggested_cable_size'])
            
            with col2:
                st.markdown("**Neutral Core**")
                st.info(results['neutral_core'])
            
            with col3:
                st.markdown("**Earth Size**")
                st.info(results['earth_size'])
            
            with col4:
                st.markdown("**Current Rating**")
                st.info(results['current_rating'])
            
            with col5:
                st.markdown("**Voltage Drop**")
                st.info(results['voltage_drop'])
            
            with col6:
                st.markdown("**Conductor Material**")
                st.info(results['conductor_material'])
                
        else:
            st.error("❌ **No suitable cable found!**")
            st.warning("Try increasing the voltage drop limit or reducing the cable length.")
            
            # Show analysis
            st.subheader("Analysis:")
            for size, data in list(calculator.cable_data.items())[:5]:
                current_rating = data['current']
                voltage_drop_per_amp_per_km = data['volt_drop']
                
                if phase == "3 Phase AC":
                    voltage_drop = load_current * (distance/1000) * voltage_drop_per_amp_per_km
                else:
                    voltage_drop = load_current * (distance/1000) * voltage_drop_per_amp_per_km * 2
                
                voltage_drop_percent = (voltage_drop / voltage) * 100
                
                st.write(f"**{size} mm²**: Current OK: {current_rating >= load_current}, Voltage Drop: {voltage_drop_percent:.1f}%")
    
    else:
        st.info("**Enter your parameters above and click 'Calculate Cable Size' to get results**")

if __name__ == "__main__":
    main()