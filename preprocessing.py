import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio


def convert_signals_to_image(ecg, pcg, duration, output_dir, filename_prefix):
    # Determine the number of samples based on duration and assumed sampling rate
    sampling_rate = 1000  # Assuming 1000 Hz sampling rate, adjust if different
    samples = int(duration * sampling_rate)

    # If the recording is longer than 30 seconds, we'll create multiple images
    if duration > 30:
        segment_duration = 30  # seconds
        segment_samples = int(segment_duration * sampling_rate)
        num_segments = samples // segment_samples

        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

            # Plot ECG segment
            ax1.plot(ecg[start:end], color='blue')
            ax1.set_title(f'ECG Signal (Segment {i+1})')
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('Amplitude')

            # Plot PCG segment
            ax2.plot(pcg[start:end], color='red')
            ax2.set_title(f'PCG Signal (Segment {i+1})')
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Amplitude')

            plt.tight_layout()

            # Save the plot as an image
            output_filename = f"{filename_prefix}_segment_{i+1}.png"
            plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Image saved as {output_filename}")
    else:
        # For 30-second recordings, create a single image
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # Plot full ECG
        ax1.plot(ecg, color='blue')
        ax1.set_title('ECG Signal')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')

        # Plot full PCG
        ax2.plot(pcg, color='red')
        ax2.set_title('PCG Signal')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')

        plt.tight_layout()

        # Save the plot as an image
        output_filename = f"{filename_prefix}.png"
        plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Image saved as {output_filename}")

def process_dataset(dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(dataset_dir):

        name = filename.replace('.mat', '')

        file=dataset_dir+str('/')+str(filename)

        if 'mat' in str(filename):
            mat_contents = sio.loadmat(file)

            chunk_size = 10000
            x = np.arange(chunk_size)

            chunks_ecg = np.array_split(mat_contents['ECG'][0],mat_contents['ECG'].shape[1]/chunk_size)
            chunks_pcg = np.array_split(mat_contents['PCG'][0],mat_contents['PCG'].shape[1]/chunk_size)

            for i in range(len(chunks_ecg)):
                plt.figure(figsize=(10, 10))

                plt.plot(x, chunks_ecg[i], color='blue')
                plt.plot(x, chunks_pcg[i], color='red')

                plt.savefig(output_dir+str("/") + name + str('_') + str(i) + '.png', format='png')

if __name__ == "__main__":
    dataset_dir = "./physionet.org/files/ephnogram/1.0.0/MAT"
    output_dir = "output"
    process_dataset(dataset_dir, output_dir)