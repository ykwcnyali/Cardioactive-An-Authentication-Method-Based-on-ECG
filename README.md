# An Authentication Method Based on ECG

## 1. Data Collecting

### I. Hardware Settings

- **EEG electrodes**

  **Red**: Right arm

  **Yellow**: Left arm

  **Green:** Right Leg

  The more electrodes close to heart, the better the quality of measuring is.

  <img src="https://pic2.zhimg.com/v2-e526379cc1ab24549d688ddae0bd6495_r.jpg" alt="img" style="zoom: 25%;" />

- **AD8232 board & Arduino Uno**

  The wiring method between AD8232 and Arduino Uno:

  | Board label | Arduino connection |
  | ----------- | ------------------ |
  | GND         | GND                |
  | 3.3V        | 3.3V               |
  | OUTPUT      | A0                 |
  | LO-         | D11                |
  | LO+         | D10                |
  | SDN         | -                  |

  <img src="https://pic2.zhimg.com/v2-b4368a3f1af6293861a3957824d07d65_r.jpg" alt="img" style="zoom:25%;" />

### II. Software Settings

- **Arduino program**

  `\Arduino\sketch_jul8a\sketch_jul8a.ino`

- **Data export program by python**

  `\ECG_supervisor.py`



## 2. Data Processing

### I. Filtering and segmentation

`"\ECG_supervisor.py"`
