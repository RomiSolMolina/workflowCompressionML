/*
 * main.c
 *
 *  Created on: Jun 16, 2023
 *      Author: ivan
 *       Brief: Assessment framework interaction with ML
 *       		inference block via ComBlock.
 *
 *       Note:	Compiler Optimization Level 3 (-O3) is mandatory
 *       		to successfully compile the code in the provided
 *       		hardware platform.
 */


#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "sleep.h"
#include "globals.h"
#include "pulses.h"



int main()
{
    init_platform();

    if(initPeripherals() == XST_SUCCESS){
    	print("Periph. initialized\n\r");

    }else{
    	print("Periph. init. failed\n\r");
    	return XST_FAILURE;
    }


    //Main loop testing the ML inference IP Core
    do{
    	print("\n\rStreaming started\n\r");

    	//Verifying all the possible class outputs
    	for(unsigned int i = 0; i < sizeof(EXPECTED_OUTPUTS)/sizeof(int); i++){

    		//Auxiliary variable to print current class type.
    		char8 outputClassChar[2];
    		outputClassChar[1] = '\0';

    		// Individual verification of each pulse type (class)
			for(unsigned int j = 0; j < PULSES_PER_TYPE; j++){
				classOutput[j] = feedPulse(PULSE_TYPE_0, j);
			}

			//Check all the ML outputs from the same class
			if(verifyOutputs(&classOutput, i) != XST_SUCCESS){
				print("Error in Class: ");
			}else{
				print("Test passed. Class: ");
			}
			outputClassChar[0] = (char8)i+48; //ASCII for class type
			print(&outputClassChar);
			print("\n\r");
    	}

    	sleep(1);

    }while(DEBUG_MODE);

    return 0;
}


/*
 * @brief	Initializes ComBlock and ML Inference IP core using the AXI4-lite interface
 * @param	None
 * @retval	Initialization status (SUCCESS or FAIL)
 */
int initPeripherals(void){
	int status = 0;

	/*
	 * ===============================
	 * Initialize Comblock peripherals
	 * ===============================
	 */

	//ML Input sample register
	cbWrite(COMBLOCK_0, CB_REG_SAMPLE, 0);

	//Clear Comblock's FIFOs
	cbWrite(COMBLOCK_0, CB_IFIFO_CONTROL, 0x01);
	cbWrite(COMBLOCK_0, CB_IFIFO_CONTROL, 0x00);

	cbWrite(COMBLOCK_0, CB_OFIFO_CONTROL, 0x01);
	cbWrite(COMBLOCK_0, CB_OFIFO_CONTROL, 0x00);


	/*
	 * ===============================
	 * Initialize ML HLS inference IP
	 * ===============================
	 */

	mlInferenceCfgPtr = XMyproject_LookupConfig(XPAR_ML_INFERENCE_DEVICE_ID);

	if(!mlInferenceCfgPtr){
		print("Lookup of ML IP failed \n\r");
		return XST_FAILURE;
	}

	status = XMyproject_CfgInitialize(&mlInferenceInstancePtr, mlInferenceCfgPtr);
	if(status != XST_SUCCESS){
		print("Couldn't init ML IP\n\r");
		return XST_FAILURE;
	}

	// We are not using interrupts in this implementation
	XMyproject_InterruptGlobalDisable(&mlInferenceInstancePtr);

	// Start IP operation with auto-restart (continuous mode)
	XMyproject_EnableAutoRestart(&mlInferenceInstancePtr);
	XMyproject_Initialize(&mlInferenceInstancePtr, XPAR_ML_INFERENCE_DEVICE_ID);
	XMyproject_Start(&mlInferenceInstancePtr);

	return XST_SUCCESS;
}


/*
 * @brief	Feeds the pre-recorded pulse shapes to the ML inference IP using
 * 			the ComBlock's output FIFO through AXI-stream. Multiplexing is carried out
 * 			by the ComBlock's CB_REG_SAMPLE register (pulse sample 0 to 29).
 * @param	pulseType:	Which type of pulse to stream (TYPE_0, TYPE_1, TYPE_2, TYPE_SAT)
 * @param	pulseIndex:	Which pulse index to send from the available pulses of a specific type
 * @retval	Class output of the ML inference block (0, 1, 2, 3)
 */
int feedPulse(unsigned int *pulseType, unsigned int pulseIndex){
	unsigned int *thisPulsePtr = pulseType + pulseIndex*SAMPLES_PER_PULSE; //Starting address of the pulse
	int readPulse = -1;
	char8 readPulseChar[2];


	//Feeding the individual pulse shape to the ML IP Core using ComBlock
	cbWrite(COMBLOCK_0, CB_OFIFO_VALUE, 0x00); //Initiazliation value for synchronization
	for(unsigned int i = 0; i < SAMPLES_PER_PULSE; i++){
		cbWrite(COMBLOCK_0, CB_OFIFO_VALUE, *(thisPulsePtr+i)); //Write to the output FIFO each pulse sample
	}

	//Reading the ML inference block output (class: 0, 1, 2, 3)
	readPulse = cbRead(COMBLOCK_0, CB_IFIFO_VALUE);

	//If UART interface is connected, print out the result in console
#if DEBUG_MODE != 0
	readPulseChar[0] = readPulse + 48; //To ASCII
	readPulseChar[1] = '\0'; //EOL
	print(&readPulseChar);
	print("\n\r");
#endif

	return readPulse;

}


/*
 * @brief	Verify that all of the outputs coincide with the expected value.
 *
 * @param	*resultsVector:	Pointer to the vector of read outputs from ML inference IP Core
 * @param	expectedOutput:	Unique expected value of the current vector list
 * @retval	Comparison result. If any incoherence is detected,
 * 			return XST_FAILURE: otherwise XST_SUCCESS
 */
int verifyOutputs(int *resultsVector, int expectedOutput){
	for(unsigned int i = 0; i < PULSES_PER_TYPE; i++){
		if (*(resultsVector + i) != expectedOutput)
			return XST_FAILURE;
	}
	return XST_SUCCESS;
}
