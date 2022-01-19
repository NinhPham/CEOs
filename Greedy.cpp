#include "Greedy.h"
#include "Utilities.h"
#include "Header.h"

/** \brief Return approximate TopK of MIPS for each query. Implements the basic Greedy from the paper NIPS 17
 - Using priority queue to find top B > K occurrences
 - Using post-processing to find top K
 *
 * \param
 *
 - MATRIX_X: point set (N x D)
 - MATRIX_Q: query set (Q x D)
 - COL_SORT_DATA_IDPAIR: vector with sorted columns (data structure used for greedy) of size N x D (col-maj)
 *
 * \return
 - Top K pair (pointIdx, dotProduct)
 *
 */
void greedy_TopK()
{
    double dStart0 = clock();

    int q, d, iSignQj, pIdx;
    double dValue = 0.0;
    double dStart = 0, dCandTime = 0, dTopKTime = 0;
    IDPair idPair;

    VectorXd vecQuery(PARAM_DATA_D); // vector of query

    vector<int> vecNextPointIdx(PARAM_DATA_D, 0); // contain the index of the point for the next verification
    vector<int> candSet; // Set with candidates already added.
    vector<bool> vecVisited(PARAM_DATA_N); // Set with candidates already added.

    priority_queue<IDPair, vector<IDPair>, less<IDPair>> candQueue; // Queue used to store candidates.
    priority_queue<IDPair, vector<IDPair>, greater<IDPair>> minQueTopK;

    vector<IDPair>::iterator iter;

    for (q = 0; q < PARAM_QUERY_Q; ++q)
    {
        dStart = clock();

        vecQuery = MATRIX_Q.col(q); // size D x 1

        candSet.clear();

        fill(vecNextPointIdx.begin(), vecNextPointIdx.end(), 0);
        fill(vecVisited.begin(), vecVisited.end(), 0);

        candQueue = priority_queue<IDPair, vector<IDPair>, less<IDPair>>();

        // Get the pointIdx with max value for each dimension
        for (d = 0; d < PARAM_DATA_D; ++d)
        {
            // First, set up vecNextPointIdx (0 if signQ < 0 else n-1)
            iSignQj = sgn(vecQuery(d));
            if (iSignQj < 0)
                vecNextPointIdx[d] = PARAM_DATA_N - 1;

            // Get the point index whose value is largest
            iter = COL_SORT_DATA_IDPAIR.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

            dValue = (*iter).m_dValue * vecQuery(d); // Value of point
            candQueue.push(IDPair(d, dValue)); // Add to queue
        }

        // Extract candidates
        while ((int)candSet.size() < PARAM_MIPS_TOP_B) // Will do at most Bd rounds
        {
            // Extract the dimension d with the max product
            d = candQueue.top().m_iIndex;
            candQueue.pop();

            // Get pointIdx and add to candidate set if not visited
            iter = COL_SORT_DATA_IDPAIR.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

            pIdx = (*iter).m_iIndex; // get index

            // If not visited
            if (!vecVisited[pIdx])
            {
                candSet.push_back(pIdx); // Add to set
                vecVisited[pIdx] = 1;
            }

            // In-/decrement counter
            iSignQj = sgn(vecQuery(d));

            //if (iSignQj == 0) // Fix bug when Qd = 0
                //iSignQj = 1;

            while (true)
            {
                vecNextPointIdx[d] += iSignQj; // next index

                // Add next element for this dimension to candQueue if any more left
                if (vecNextPointIdx[d] >= 0 && vecNextPointIdx[d] < PARAM_DATA_N)
                {
                    iter = COL_SORT_DATA_IDPAIR.begin() + vecNextPointIdx[d] + d * PARAM_DATA_N;

                    pIdx = (*iter).m_iIndex; // Point index of next

                    if (!vecVisited[pIdx]) // if not exist
                    {
                        dValue = (*iter).m_dValue * vecQuery(d); // Value of next
                        candQueue.push(IDPair(d, dValue));      // Add to queue

                        break;
                    }
                }
                else
                    break;
            }

        }

        dCandTime += clock() - dStart;

        // Post computation
        dStart = clock();

        minQueTopK = priority_queue<IDPair, vector<IDPair>, greater<IDPair>>(); //priority_queue does not have clear()
        extract_TopK_MIPS(vecQuery, candSet, PARAM_MIPS_TOP_K, minQueTopK);

        dTopKTime += clock() - dStart;

        if (PARAM_INTERNAL_SAVE_OUTPUT)
            saveQueue(minQueTopK, "greedy_TopK_Post_" + int2str(q) + ".txt");
    }

    // Print time complexity of each step
    printf("Time for generating candidate set %f \n", getCPUTime(dCandTime));
    printf("TopK time is %f \n", getCPUTime(dTopKTime));

    printf("Greedy: Time is %f \n", getCPUTime(clock() - dStart0));
}
