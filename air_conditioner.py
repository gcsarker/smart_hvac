import numpy as np

class AirConditioner():
    def __init__(self):
        self.room_height = 5
        self.room_width = 6
        self.room_length = 6
        self.floor_area = self.room_length*self.room_width # (m2)
        self.room_volume = self.floor_area*self.room_height # (m3)
        self.wall_area = self.room_width*self.room_height # (m2)
        self.window_area = 2*2 # 2m by 2m window


        self.Specific_heat_air = 1.005 # Specific heat of air (Kj/Kg-k)
        self.air_density = 1.204 # Air density (Kg/m3)
        self.latent_heat_vaporization = 2430 # latent heat of vaporization (Kj/kg)
        self.P_atm = 101.325 # Atmospheric pressure (KPa)

        self.number_of_persons = 2
        self.air_flow_rate = self.number_of_persons*(17*0.0004719) # air flow rate for ventillation (m3/s)

        self.U = 2 # Heat transfer coefficient of concrete wall (W/(m2K)) for 10inch concrete wall
        # we can also consider U = 3.9 for 6inch concrete wall
        self.U_window = 1.6 # (W/m2K), Around 5 for single glaze window
        self.SHGC = 0.5 # Solar heat gain coefficient. provides the fraction of shading
        self.I_solar = 150 # Solar Iradiation on wall (W/m2), taken as constant for simplicity
        self.Absorptivity = 0.4 # Iradiation apsorption ability percentage of black body absorption
        self.H = 20 # Convective heat transfer coefficient of outside concrete wall (W/m2-K)
        # typical values between (15-25)

        self.Sensible_heat_gain_human = 75 # (W)
        self.latent_heat_gain_human = 55 # (W) 

        self.COP_rated = 4.25 # Coefficient of performance rated by air conditioner 
        # Calculated by (Cooling capacity/ rated power)
        self.Cooling_capacity = 5.25 # KW

    def calculate_ventilation_sensible_cooling_load(self, T_indoor, T_outdoor):
        """
        sensible cooling load for ventilation
        Q_s = air_flow_rate*air_density*Specific_heat_air(T_outdoor - T_indoor)

        Kj/s or KW = (m3/s)*(Kj / Kg.K)*(Kg/m3)(K)

        INPUT
            T_outdoor = Outdoor temperature (degree C / K)
            T_indoor = Indoor Temperature (degree C / K)

        OUTPUT
            Q_s = Sensible heat load for ventilation (KW)

        """
        Q_s = self.air_flow_rate*self.air_density*self.Specific_heat_air*(T_outdoor - T_indoor)
        return Q_s

    def calculate_ventilation_latent_cooling_load(self, RH_indoor, RH_outdoor, T_indoor, T_outdoor):
        """
        Latent Cooling load for ventilation
        Q_l = air_flow_rate*air_density*latent_heat_vaporization(w2 - w1)
        w2 = outdoor water vapor to dry air ratio
        w1 = indoor water vapor to dry air ratio

        w = 0.622 * (relative_humidity * P_sat)/(P_atm - relative_humidity*P_sat)
        P_sat = 0.61078 * e(17.27*T/(T+237.3))

        INPUT
            RH_indoor = indoor relative_humidity (%)
            RH_outdoor = outdoor relative humidity (%)
            T_indoor = indoor temperature (degree celcius)
            T_outdoor = outdoor temperature (degree celcius)
        
        OUTPUT
            Q_l
        """
        RH_indoor = RH_indoor/100
        RH_outdoor = RH_outdoor/100
        P_sat_indoor = 0.61078 * np.exp(17.27*T_indoor/(T_indoor+237.3))
        P_sat_outdoor = 0.61078 * np.exp(17.27*T_outdoor/(T_outdoor+237.3))
        w1 = 0.622 * (RH_indoor * P_sat_indoor)/(self.P_atm - RH_indoor*P_sat_indoor)
        w2 = 0.622 * (RH_outdoor * P_sat_outdoor)/(self.P_atm - RH_outdoor*P_sat_outdoor)
        Q_l = self.air_flow_rate*self.air_density*self.latent_heat_vaporization*(w2-w1)
        return Q_l


    def calculate_heat_gain_from_wall(self, T_indoor, T_outdoor):
        """ 
        Heat transfer through wall
        Q_wall = U*wall_area*(T_outdoor_surface - T_indoor)
        U = Overall heat transfer coefficient of wall
        wall_area = surface area of the wall
        T_outdoor_surface = Outdoor wall surface temperature
        T_indoor = indoor temperature

        T_outdoor_surface = T_outdoor + (I_solar * Absorptivity)/H
        I_solar = solar irradiation on wall surface
        Absorptivity = Absorption coefficient of wall
        H = Convective heat transfer coefficient.

        INPUT:
            T_indoor = indoor temperature (degree celcius)
            T_outdoor = outdoor temperature (degree celcius)

        OUTPUT:
            Q_wall (Kw)

        """
        T_outdoor_surface = T_outdoor + (self.I_solar * self.Absorptivity)/self.H
        Q_wall = self.U*self.wall_area*(T_outdoor_surface - T_indoor)*0.001 
        return Q_wall

    def calculate_heat_gain_from_window(self, T_indoor, T_outdoor):
        """ 
        Heat gain through Windows

        Q_window = U_window*window_area*(T_outdoor - T_indoor)
        U_window = Overall heat transfer coefficient of glass window
        window_area = surface area of the wall
        T_outdoor = Outdoor temperature
        T_indoor = indoor temperature

        INPUT:
            T_indoor = indoor temperature
            T_outdoor = outdoor temperature

        OUTPUT:
            Q_window

        """
        Q_window = self.U_window*self.window_area*(T_outdoor - T_indoor)
        Q_window_solar = self.window_area*self.SHGC*self.I_solar

        return Q_window + Q_window_solar
    
    def calculate_total_heat_gain(self, T_indoor, T_outdoor, RH_indoor, RH_outdoor):
        # Cooling loads
        Q_s = self.calculate_ventilation_sensible_cooling_load(T_indoor, T_outdoor)
        Q_l = self.calculate_ventilation_latent_cooling_load(RH_indoor, RH_outdoor, T_indoor, T_outdoor)
        Q_wall = self.calculate_heat_gain_from_wall(T_indoor, T_outdoor)
        # Q_window = calculate_heat_gain_from_window(T_indoor, T_outdoor)
        Q_appliances = 10.8*self.floor_area * 0.001 # heat gain from appliances (KW)
        Q_humans = self.number_of_persons* (self.Sensible_heat_gain_human + self.latent_heat_gain_human) * 0.001 # heat gain from humans (KW)
        Q_light = 30 * 0.001 # lighting heat gain (KW/m2) 
        Q_aux = 100 * 0.001 # extra energy to used in compressor, fan and electronics (KW)
        Q_total = Q_s + Q_l + Q_wall + Q_appliances + Q_humans + Q_light + Q_aux
        return Q_total
    
    def calculate_energy_consumption(self, T_indoor, T_outdoor, RH_indoor, RH_outdoor, T_setpoint):
        # if T_indoor - T_setpoint < 2:
        #     Q_total = self.calculate_total_heat_gain(T_indoor, T_outdoor, RH_indoor, RH_outdoor)

        #     COP_dyn = self.COP_rated - 0.1*np.abs(T_indoor-T_outdoor)
        #     Power = Q_total/COP_dyn # KW
        # else:
        #     Power = self.Cooling_capacity

        Q_total = self.calculate_total_heat_gain(T_indoor, T_outdoor, RH_indoor, RH_outdoor)
        COP_dyn = self.COP_rated - 0.1*np.abs(T_indoor-T_outdoor)
        Power = Q_total/COP_dyn # KW  
              
        # Energy = Power*time
        return Power
    
    def calculate_room_temperature_after_cooling(self, Q_gain, T_init, time = 60):
        Q_net = Q_gain - self.Cooling_capacity
        delta_T = (Q_net / (self.air_density*self.room_volume*self.Specific_heat_air))
        delta_T = delta_T*time
        return T_init + delta_T
    
    def calculate_room_temperature_when_standby(self, Q_gain, T_init, time = 60):
        delta_T = (Q_gain / (self.air_density*self.room_volume*self.Specific_heat_air))
        delta_T = delta_T*time
        return T_init + delta_T
