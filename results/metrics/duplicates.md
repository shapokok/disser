# PlantVillage train/valid leakage check

Hash: 64-bit dHash, near-duplicate threshold Hamming ≤ 4.

* valid images with a near-duplicate in train: **385 of 17572 (2.2 %)**
* exact hash matches: 42; duplicates labelled with a different class: 25

| Class | valid | leaked | share |
|---|---|---|---|
| Potato___healthy | 456 | 96 | 21.1 % |
| Apple___Cedar_apple_rust | 440 | 80 | 18.2 % |
| Raspberry___healthy | 445 | 30 | 6.7 % |
| Tomato___Target_Spot | 457 | 25 | 5.5 % |
| Tomato___Tomato_mosaic_virus | 448 | 19 | 4.2 % |
| Tomato___healthy | 481 | 20 | 4.2 % |
| Peach___healthy | 432 | 15 | 3.5 % |
| Cherry_(including_sour)___healthy | 456 | 10 | 2.2 % |
