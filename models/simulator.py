import numpy as np
import torch
import math


class Simulator:
    def __init__(self, cycle_len=257, device_name='cpu'):
        self.A = torch.tensor(0.005).to(device_name)  # mV
        self.f1 = torch.tensor(0.1).to(device_name)  # mean 1
        self.f2 = torch.tensor(0.25).to(device_name)  # mean 2
        self.c1 = torch.tensor(0.01).to(device_name)  # std 1
        self.c2 = torch.tensor(0.01).to(device_name)  # std 2
        self.cycle_len = cycle_len
        self.h = torch.tensor(1 / cycle_len).to(device_name)
        self.rrpc = self.generate_omega_function()
        self.rrpc = torch.tensor(self.rrpc).to(device_name).float()
        self.TYPICAL_ODE_N_PARAMS = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                        30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                        160.0 * math.pi / 180.0]

        self.TYPICAL_ODE_L_PARAMS = [0.2, 0.25, -0.5 * math.pi, -1.0, 0.1, -15.0 * math.pi / 180.0,
                                30.0, 0.1, 0.0 * math.pi / 180.0, -10.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                                160.0 * math.pi / 180.0]

        self.TYPICAL_ODE_R_PARAMS = [0.8, 0.25, -0.5 * math.pi, -10.0, 0.1, -15.0 * math.pi / 180.0,
                                30.0, 0.1, 0.03 * math.pi / 180.0, -10.0, 0.1, 15.0 * math.pi / 180.0, 0.5, 0.2,
                                160.0 * math.pi / 180.0]

        self.TYPICAL_ODE_V_PARAMS = [0.1, 0.6, -0.5 * math.pi,
                                0.0, 0.1, -15.0 * math.pi / 180.0,
                                30.0, 0.1, 0.00 * math.pi / 180.0,
                                -10.0, 0.1, 15.0 * math.pi / 180.0,
                                0.5, 0.2, 160.0 * math.pi / 180.0]
        #
        # Helper dict:
        #
        self.beat_type_to_typical_param = {'N': self.TYPICAL_ODE_N_PARAMS, 'L': self.TYPICAL_ODE_L_PARAMS, 'R': self.TYPICAL_ODE_R_PARAMS,
                                    'V': self.TYPICAL_ODE_V_PARAMS}

    def generate_omega_function(self):
        """
        :return:
        """
        rr = self.rrprocess(self.cycle_len, lfhfratio=0.5, hrmean=60, hrstd=1, sf=512)
        return rr

    def rrprocess(self, n, lfhfratio=0.5, hrmean=60, hrstd=1, sf=512):
        """
        GENERATE RR PROCESS
        :param flo:
        :param fhi:
        :param flostd:
        :param fhistd:
        :param lfhfratio:
        :param hrmean:
        :param hrstd:
        :param sf:
        :param n:
        :return:
        """
        # Step 1: Calculate the power spectrum:
        amplitudes = np.linspace(0, 1, n)
        # phases = np.random.normal(loc=0, scale=1, size=len(S_F)) * 2 * np.pi
        phases = np.linspace(0, 1, n) * 2 * np.pi
        complex_series = [complex(amplitudes[i] * np.cos(phases[i]), amplitudes[i] * np.sin(phases[i])) for i in
                        range(len(phases))]

        T = np.fft.ifft(complex_series, n)
        T = T.real

        rrmean = 60.0 / hrmean
        rrstd = 60.0 * hrstd / (hrmean * hrmean)

        std = np.std(T)
        ratio = rrstd / std
        T = ratio * T
        T = T + rrmean
        return T

    def generate_typical_N_ode_params(self, b_size, device):
        noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
        params = 0.1 * noise_param + torch.Tensor(self.TYPICAL_ODE_N_PARAMS).to(device)
        return params


    def generate_typical_L_ode_params(self, b_size, device):
        noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
        params = 0.1 * noise_param + torch.Tensor(self.TYPICAL_ODE_L_PARAMS).to(device)
        return params


    def generate_typical_R_ode_params(self, b_size, device):
        noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
        params = 0.1 * noise_param + torch.Tensor(self.TYPICAL_ODE_R_PARAMS).to(device)
        return params

    def generate_typical_V_ode_params(self, b_size, device):
        noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
        params = 0.1 * noise_param + torch.Tensor(self.TYPICAL_ODE_V_PARAMS).to(device)
        return params

    
    def d_x_d_t(self, y, x, t, rrpc, delta_t):
        alpha = 1 - ((x * x) + (y * y)) ** 0.5

        cast = (t / delta_t).type(torch.IntTensor)
        tensor_temp = 1 + cast
        tensor_temp = tensor_temp % len(rrpc)
        if rrpc[tensor_temp] == 0:
            omega = (2.0 * math.pi / 1e-3)
        else:
            omega = (2.0 * math.pi / rrpc[tensor_temp])

        f_x = alpha * x - omega * y
        return f_x


    def d_y_d_t(self, y, x, t, rrpc, delta_t):
        alpha = 1 - ((x * x) + (y * y)) ** 0.5

        cast = (t / delta_t).type(torch.IntTensor)
        tensor_temp = 1 + cast
        tensor_temp = tensor_temp % len(rrpc)
        if rrpc[tensor_temp] == 0:
            omega = (2.0 * math.pi / 1e-3)
        else:
            omega = (2.0 * math.pi / rrpc[tensor_temp])

        f_y = alpha * y + omega * x
        return f_y

    def d_z_d_t(self, x, y, z, t, params):
        """

        :param x:
        :param y:
        :param z:
        :param t:
        :param params:
        :param ode_params: Nx15
        :return:
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = 'cpu'
        # A = 
        # f2 = 
        a_p, a_q, a_r, a_s, a_t = params[:, 0], params[:, 3], params[:, 6], params[:, 9], params[:, 12]

        b_p, b_q, b_r, b_s, b_t = params[:, 1], params[:, 4], params[:, 7], params[:, 10], params[:, 13]

        theta_p, theta_q, theta_r, theta_s, theta_t = params[:, 2], params[:, 5], params[:, 8], params[:, 11], params[:, 14]

        a_p = a_p.view(-1, 1)
        a_q = a_q.view(-1, 1)
        a_r = a_r.view(-1, 1)
        a_s = a_s.view(-1, 1)
        a_t = a_t.view(-1, 1)

        b_p = b_p.view(-1, 1)
        b_q = b_q.view(-1, 1)
        b_r = b_r.view(-1, 1)
        b_s = b_s.view(-1, 1)
        b_t = b_t.view(-1, 1)

        theta_p = theta_p.view(-1, 1)
        theta_q = theta_q.view(-1, 1)
        theta_r = theta_r.view(-1, 1)
        theta_s = theta_s.view(-1, 1)
        theta_t = theta_t.view(-1, 1)

        theta = torch.atan2(y, x)
        delta_theta_p = torch.fmod(theta - theta_p, 2 * math.pi)
        delta_theta_q = torch.fmod(theta - theta_q, 2 * math.pi)
        delta_theta_r = torch.fmod(theta - theta_r, 2 * math.pi)
        delta_theta_s = torch.fmod(theta - theta_s, 2 * math.pi)
        delta_theta_t = torch.fmod(theta - theta_t, 2 * math.pi)

        z_p = a_p * delta_theta_p * \
            torch.exp((- delta_theta_p * delta_theta_p / (2 * b_p * b_p)))

        z_q = a_q * delta_theta_q * \
            torch.exp((- delta_theta_q * delta_theta_q / (2 * b_q * b_q)))

        z_r = a_r * delta_theta_r * \
            torch.exp((- delta_theta_r * delta_theta_r / (2 * b_r * b_r)))

        z_s = a_s * delta_theta_s * \
            torch.exp((- delta_theta_s * delta_theta_s / (2 * b_s * b_s)))

        z_t = a_t * delta_theta_t * \
            torch.exp((- delta_theta_t * delta_theta_t / (2 * b_t * b_t)))

        z_0_t = (self.A * torch.sin(2 * math.pi * self.f2 * t))

        z_p = z_p.to(device)
        z_q = z_q.to(device)
        z_r = z_r.to(device)
        z_s = z_s.to(device)
        z_t = z_t.to(device)
        z_0_t = z_0_t.to(device)

        f_z = -1 * (z_p + z_q + z_r + z_s + z_t) - (z - z_0_t)
        return f_z