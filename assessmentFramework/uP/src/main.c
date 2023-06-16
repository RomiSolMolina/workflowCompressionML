/*
 * main.c
 *
 *  Created on: Jun 16, 2023
 *      Author: ivan
 *       Brief: Assessment framework interaction with ML
 *       		inference block via ComBlock
 */


#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
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


    //Only testing a single pulse
    feedPulse(PULSE_TYPE_0, 0);


    cleanup_platform();
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
 * @retval	Data streaming status (SUCCESS or FAIL)
 */
int feedPulse(unsigned int *pulseType, unsigned int pulseIndex){
	unsigned int *thisPulsePtr = pulseType + pulseIndex*SAMPLES_PER_PULSE; //Starting address of the pulse

	cbWrite(COMBLOCK_0, CB_OFIFO_VALUE, 0x00); //Initiazliation value for synchronization

	for(unsigned int i = 0; i < SAMPLES_PER_PULSE; i++){
		cbWrite(COMBLOCK_0, CB_OFIFO_VALUE, *(thisPulsePtr+i)); //Write to the output FIFO each pulse sample
	}

	return XST_SUCCESS;


}
