import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from Networks import alice, bob, eve, abemodel, m_train, m_bits, evemodel
from EllipticCurvesExtended import EllipticCurve,curve

# Setup elliptic curve
i = 5


# Create output directories
curve_dir = f'curve_{curve.a}_{curve.b}_{curve.p}'
results_dir = os.path.join(curve_dir, f'{1}cycle')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)

# Initialize loss trackers
evelosses = []
boblosses = []
abelosses = []

# Training settings
n_epochs = 20  
batch_size = 64
n_batches = m_train // batch_size  
abecycles = 5
evecycles = 1

epoch = 0
start = time.time()
while epoch < n_epochs:
    print(f"\n=== Starting Epoch {epoch} ===")  # Debug
    evelosses0 = []
    boblosses0 = []
    abelosses0 = []
    for iteration in range(n_batches):
        # A-B Training
        alice.trainable = True
        for cycle in range(abecycles):
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits).astype(np.float32)
            private_arr, public_arr = curve.generate_keypair(batch_size)
            y_true = [m_batch, m_batch]
            loss = abemodel.train_on_batch([m_batch, public_arr, private_arr], y_true)

        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Bob Evaluation
        m_enc = alice.predict([m_batch, public_arr])
        m_dec = bob.predict([m_enc, private_arr])
        loss_bob_eval = np.mean(np.sum(np.abs(m_batch - m_dec), axis=-1))
        boblosses0.append(loss_bob_eval)
        boblosses.append(loss_bob_eval)
        bobavg = np.mean(boblosses0)

        # Eve Training
        alice.trainable = False
        for cycle in range(evecycles):
            m_batch_eve = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits).astype(np.float32)
            _, public_arr_eve = curve.generate_keypair(batch_size)
            y_true_eve = m_batch_eve
            loss_eve_train = evemodel.train_on_batch([m_batch_eve, public_arr_eve], y_true_eve)

        evelosses0.append(loss_eve_train)
        evelosses.append(loss_eve_train)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    epoch += 1

print("\nTraining complete.")
end = time.time()
print("Elapsed time:", end - start)

# Save losses to CSV
steps = -1
Biodata = {
    'ABloss': abelosses[:steps],
    'Bobloss': boblosses[:steps],
    'Eveloss': evelosses[:steps]
}
df = pd.DataFrame(Biodata)
df.to_csv(os.path.join(results_dir, f'test-{i}.csv'), mode='a', index=False)

# Plot losses
plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'figures', f'restult-{i}.png'))

# Save test results to text file
with open('results.txt', "a") as f:
    f.write("Training complete.\n")
    f.write(f"Curve: a={curve.a}, b={curve.b}, p={curve.p}\n")
    f.write("Epochs: {}\n".format(n_epochs))
    f.write("Batch size: {}\n".format(batch_size))
    f.write("Iterations per epoch: {}\n".format(n_batches))
    f.write("Alice-Bob cycles per iteration: {}\n".format(abecycles))
    f.write("Eve cycles per iteration: {}\n".format(evecycles))

    # Final accuracy evaluation
    print("Starting final evaluation...")  # Debug
    m_batch_final = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits).astype(np.float32)
    private_arr_final, public_arr_final = curve.generate_keypair(batch_size)

    cipher = alice.predict([m_batch_final, public_arr_final])
    decrypted = bob.predict([cipher, private_arr_final])
    decrypted_bits = np.round(decrypted).astype(int)
    correct_bits = np.sum(decrypted_bits == m_batch_final)
    total_bits = np.prod(decrypted_bits.shape)
    accuracy = correct_bits / total_bits * 100

    eve_decrypted = eve.predict(cipher)
    eve_decrypted_bits = np.round(eve_decrypted).astype(int)
    correct_bits_eve = np.sum(eve_decrypted_bits == m_batch_final)
    accuracy_eve = correct_bits_eve / total_bits * 100

    print(f"Number of correctly decrypted bits by Bob: {correct_bits}")
    print(f"Decryption accuracy by Bob: {accuracy:.2f}%")
    print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
    print(f"Decryption accuracy by Eve: {accuracy_eve:.2f}%")

    f.write(f"Total number of bits: {total_bits}\n")
    f.write(f"Number of correctly decrypted bits by Bob: {correct_bits}\n")
    f.write(f"Decryption accuracy by Bob: {accuracy:.2f}%\n")
    f.write(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}\n")
    f.write(f"Decryption accuracy by Eve: {accuracy_eve:.2f}%\n")
    f.write("\n")