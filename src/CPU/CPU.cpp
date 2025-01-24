#include <cstring>
#include <vector>

#include "CPU/CPU.hpp"

namespace CPU 
{
    void fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter) 
    {

    }

    void fixed_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
    {
        // works only for some inputs
        output_size = 0;
        output = nullptr;
        long unsigned int max_portion_size = 1 << 25;

        long unsigned int processed_size = 0;
        unsigned int portion_size = 0;
        std::memcpy(&portion_size, input, 4);
        input += 4;

		std::vector<unsigned char> temp_output = std::vector<unsigned char>();

        while (processed_size < input_size)
        {
            long unsigned int portion_output_size = 0;

            unsigned int frame_count = 0;
            unsigned int frame_size_B = 0;
            std::memcpy(&frame_count, input, 4);
            std::memcpy(&frame_size_B, input + 4, 2);
            unsigned int frame_size_b = frame_size_B * 8;
            unsigned int current_bit_offset = 0;

			input += 6;
            unsigned char* array_sizes = input;
			unsigned char* array_zeros = input + frame_count * 2;
			unsigned char* data = input + frame_count * 4;

            unsigned char* frame_out = new unsigned char[frame_size_B];

            for (unsigned int frame_num = 0; frame_num < frame_count; frame_num++)
            {
                unsigned int comp_seg_size = array_sizes[frame_num] + (array_sizes[frame_num + 1] << 8);
                unsigned int insig_zeros = array_zeros[frame_num] + (array_zeros[frame_num + 1] << 8);
				array_sizes += 2;
				array_zeros += 2;

                unsigned int seg_size = comp_seg_size + insig_zeros;
                unsigned int seg_count = frame_size_b / seg_size;

                unsigned int comp_frame_length_b = seg_count * comp_seg_size;
                unsigned int remainder_size = frame_size_b % seg_size;
                if (remainder_size != 0) {
                    comp_frame_length_b += remainder_size - insig_zeros;
                }

				std::memset(frame_out, 0, frame_size_B);
				unsigned int frame_out_offset = insig_zeros;

				unsigned int bytes_to_read = (current_bit_offset + comp_frame_length_b) / 8 + ((current_bit_offset + comp_frame_length_b) % 8 != 0);
                unsigned int bit_position = current_bit_offset;
				unsigned int bit_num = 0;
				for (unsigned int byte_num = 0; byte_num < bytes_to_read; byte_num++)
				{
					unsigned char byte = data[byte_num];
                    while (bit_position < 8 && bit_num < comp_frame_length_b) {
                        unsigned char bit = (byte >> (7 - bit_position)) & 1;

                        frame_out[frame_out_offset / 8] |= bit << (7 - (frame_out_offset % 8));
                        frame_out_offset++;
                        bit_position++;
                        bit_num++;

                        if (bit_num % comp_seg_size == 0) {
                            frame_out_offset += insig_zeros;
                        }

                        current_bit_offset++;
                        if (current_bit_offset == 8) {
                            current_bit_offset = 0;
                        }


                    }
					bit_position = 0;
				}

				temp_output.insert(temp_output.end(), frame_out, frame_out + frame_size_B);
				data += bytes_to_read;
				if (current_bit_offset != 0) {
					data--;
				}
            }

            delete[] frame_out;
            processed_size += 4 + portion_size;
            input += portion_size;
            std::memcpy(&portion_size, input, 4);
            input += 4;
        }

		output_size = temp_output.size();
		output = new unsigned char[output_size];
		std::copy(temp_output.begin(), temp_output.end(), output);
    }

    void run_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter)
    {
        unsigned int symbol_size = parameter;
        unsigned int partition_size = 255;

        std::vector<unsigned char> array = std::vector<unsigned char>();
        std::vector<unsigned char> data = std::vector<unsigned char>();
        int count = 1;
        std::vector<unsigned char> last_symbol = std::vector<unsigned char>();
        std::vector<unsigned char> this_symbol = std::vector<unsigned char>();

        for (long unsigned int byte = 0; byte < input_size; byte++)
        {
            if (last_symbol.size() < symbol_size)
            {
                last_symbol.push_back(input[byte]);
                continue;
            }

            if (this_symbol.size() < symbol_size)
            {
                this_symbol.push_back(input[byte]);
                continue;
            }

            if (last_symbol != this_symbol) {
                if (count != 0) {
                    array.push_back(count);
                    for (unsigned char& symbol : last_symbol)
                    {
                        data.push_back(symbol);
                    }
                }

                std::copy(this_symbol.begin(), this_symbol.end(), last_symbol.begin());
                this_symbol.clear();
                this_symbol.push_back(input[byte]);
                count = 1;
                continue;
            }

            count++;
            if (count == partition_size)
            {
                array.push_back(count);
                for (unsigned char& symbol : last_symbol)
                {
                    data.push_back(symbol);
                }
                count = 0;
            }
            this_symbol.clear();
			this_symbol.push_back(input[byte]);
        }

		if (last_symbol == this_symbol)
		{
			count++;
			array.push_back(count);
			for (unsigned char& symbol : last_symbol)
			{
				data.push_back(symbol);
			}
		}
		else
		{
			if (count != 0)
			{
                array.push_back(count);
                for (unsigned char& symbol : last_symbol)
                {
                    data.push_back(symbol);
                }
			}

			if (this_symbol.size() == symbol_size)
			{
                array.push_back(1);
			}

            for (unsigned char& symbol : this_symbol)
            {
                data.push_back(symbol);
            }
		}

		output_size = array.size() + data.size() + 5;
		unsigned int array_size = array.size();
		output = new unsigned char[output_size];
		output[0] = symbol_size;
        std::memcpy(output+1, &array_size, 4);
		std::memcpy(output + 5, array.data(), array.size());
		std::memcpy(output + 5 + array.size(), data.data(), data.size());
    }

    void run_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size)
    {
        unsigned int header_size = 5;
        unsigned int symbol_size = input[0];
        unsigned int array_size;
        std::memcpy(&array_size, input + 1, 4);
        unsigned int remaining_symbols = input_size - header_size - array_size - array_size * symbol_size;

        output_size = 0;
        output = new unsigned char[symbol_size * array_size * 255];
        for (unsigned int symbol_index = 0; symbol_index < array_size; ++symbol_index)
        {
            for (int rep = 0; rep < input[header_size + symbol_index]; ++rep)
            {
                for (int byte = 0; byte < symbol_size; ++byte)
                {
                    output[output_size] = input[header_size + array_size + symbol_index * symbol_size + byte];
                    output_size++;
                }
            }
        }
        while (remaining_symbols > 0)
        {
            output[output_size++] = input[input_size - remaining_symbols--];
        }
    }
}

