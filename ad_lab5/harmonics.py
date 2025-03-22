import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Button, CheckboxGroup, Select
from scipy.signal import butter, filtfilt


def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise):
    t = np.linspace(0, 2 * np.pi, 500)  # Time axis
    y_harmonic = amplitude * np.sin(frequency * t + phase)  # Harmonic signal

    if show_noise:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), len(t))  # Gaussian noise
        y_noisy = y_harmonic + noise
    else:
        y_noisy = y_harmonic

    return t, y_harmonic, y_noisy


def moving_average_filter(signal, window_size):
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")

    # Pad the signal at the edges to handle boundary effects
    half_window = window_size // 2
    padded_signal = np.pad(signal, (half_window, half_window), mode='edge')

    # Apply the moving average filter
    filtered_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')

    return filtered_signal


def apply_iir_filter(signal, cutoff_frequency, sampling_rate, order=4):
    # Normalize the cutoff frequency (Nyquist frequency = sampling_rate / 2)
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply the filter using zero-phase filtering (filtfilt)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


# Default parameters
amplitude = 1.0
frequency = 1.0
phase = 0.0
noise_mean = 0.0
noise_covariance = 0.1
show_noise_flag = True

# Generate initial data
t, y_harmonic, y_noisy = harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise_flag)

# Create a ColumnDataSource for Bokeh
source = ColumnDataSource(data=dict(t=t, y_harmonic=y_harmonic, y_noisy=y_noisy, y_filtered=y_noisy))

# Create the plot
plot = figure(
    title="Harmonic Signal with Noise",
    x_axis_label='Time',
    y_axis_label='Amplitude',
    width=800,
    height=400,
    background_fill_color="#18314F",
    border_fill_color="#BD8B9C"
)
plot.line('t', 'y_harmonic', source=source, legend_label="Harmonic", line_width=2, color="#8BBEB2")
plot.line('t', 'y_noisy', source=source, legend_label="Noisy Signal", line_width=2, color="#AF125A",
          visible=show_noise_flag)
plot.line('t', 'y_filtered', source=source, legend_label="Filtered Signal", line_width=2, color="#E6F9AF",
          visible=False)

# Sliders
amplitude_slider = Slider(start=0.1, end=5.0, value=amplitude, step=0.1, title="Amplitude")
frequency_slider = Slider(start=0.1, end=5.0, value=frequency, step=0.1, title="Frequency")
phase_slider = Slider(start=0, end=2 * np.pi, value=phase, step=0.1, title="Phase")
noise_mean_slider = Slider(start=-1.0, end=1.0, value=noise_mean, step=0.1, title="Noise Mean")
noise_covariance_slider = Slider(start=0.01, end=1.0, value=noise_covariance, step=0.01, title="Noise Covariance")

# Dropdown menu for filter selection
filter_select = Select(
    title="Filter",
    value="No Filter",
    options=["No Filter", "Moving Average", "IIR Filter"]
)

# Slider for cutoff frequency
cutoff_slider = Slider(start=0.1, end=5.0, value=1.0, step=0.1, title="Cutoff Frequency (Hz)")

# Slider for filter order
order_slider = Slider(start=1, end=10, value=4, step=1, title="Filter Order")

# Checkbox for showing noise
checkbox = CheckboxGroup(labels=["Show Noise"], active=[0] if show_noise_flag else [])

# Reset button
reset_button = Button(label="Reset", button_type="success")


def update_plot(attr, old, new):
    global show_noise_flag
    amplitude = amplitude_slider.value
    frequency = frequency_slider.value
    phase = phase_slider.value
    noise_mean = noise_mean_slider.value
    noise_covariance = noise_covariance_slider.value
    show_noise_flag = 0 in checkbox.active  # Check if "Show Noise" is selected

    # Generate updated data
    t, y_harmonic, y_noisy = harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance,
                                                 show_noise_flag)

    # Apply the selected filter
    filter_type = filter_select.value
    if filter_type == "Moving Average":
        window_size = 11  # You can make this adjustable with another slider if needed
        y_filtered = moving_average_filter(y_noisy, window_size)
        filtered_visible = True
    elif filter_type == "IIR Filter":
        sampling_rate = len(t) / (t[-1] - t[0])  # Calculate sampling rate
        cutoff_frequency = cutoff_slider.value
        order = int(order_slider.value)
        y_filtered = apply_iir_filter(y_noisy, cutoff_frequency, sampling_rate, order)
        filtered_visible = True
    else:
        y_filtered = y_noisy  # No filtering
        filtered_visible = False

    # Update the data source
    source.data = dict(t=t, y_harmonic=y_harmonic, y_noisy=y_noisy, y_filtered=y_filtered)

    # Toggle visibility of the noisy and filtered signals
    plot.renderers[1].visible = show_noise_flag  # Noisy signal
    plot.renderers[2].visible = filtered_visible  # Filtered signal


def update_filter_controls(filter_type):
    if filter_type == "IIR Filter":
        cutoff_slider.disabled = False
        order_slider.disabled = False
    else:
        cutoff_slider.disabled = True
        order_slider.disabled = True


def filter_select_callback(attr, old, new):
    # Update the plot when the filter type changes
    update_plot(attr, old, new)

    # Enable or disable the cutoff and order sliders
    update_filter_controls(new)


for widget in [amplitude_slider, frequency_slider, phase_slider, noise_mean_slider, noise_covariance_slider,
               cutoff_slider, order_slider]:
    widget.on_change('value', update_plot)

checkbox.on_change('active', update_plot)
filter_select.on_change('value', filter_select_callback)


def reset_parameters():
    amplitude_slider.value = 1.0
    frequency_slider.value = 1.0
    phase_slider.value = 0.0
    noise_mean_slider.value = 0.0
    noise_covariance_slider.value = 0.1
    checkbox.active = [0]  # Show noise by default
    filter_select.value = "No Filter"
    cutoff_slider.value = 1.0
    order_slider.value = 4


reset_button.on_click(reset_parameters)

# Initialize slider states based on the default filter type
update_filter_controls(filter_select.value)

layout = column(
    row(plot),
    column(
        amplitude_slider,
        frequency_slider,
        phase_slider,
        noise_mean_slider,
        noise_covariance_slider,
        checkbox,
        reset_button,
        filter_select,
        cutoff_slider,
        order_slider
    )
)

curdoc().add_root(layout)
curdoc().title = "Interactive Harmonic Signal"