#pragma once

#define DLL_EXPORT __declspec(dllexport)

extern "C"
{
	int DLL_EXPORT SimpleReturnFunc();
}
